"""The local LLM agent: SmolLM2-135M-Instruct on CPU.

A 135M model is unreliable at free generation, so it never generates. It *scores*: one batched
forward pass rates every option and the agent takes the argmax. Methods:

* ``ask``: chat-format the question, then score each word as the continuation of
  "The odd one out is". Mean log-prob per token, so long words aren't penalised.
* ``coherence``: leave-one-out. For each candidate, score how natural the *other four* read as
  a list of related words. Removing the intruder leaves the most coherent list.
* ``both``: sum of the two methods' z-scored ranks.

``tools/calibrate_agent.py`` measures each method on fresh puzzles; DEFAULT_METHOD is set from it.
"""

from __future__ import annotations

import math
import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("AGENT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DEFAULT_METHOD = os.environ.get("AGENT_METHOD", "coherence")

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


class Agent:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.lock = threading.Lock()  # one forward pass at a time on CPU
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def _continuation_logprobs(self, prefixes: list[str], conts: list[str],
                               trace: list | None = None) -> list[float]:
        """Mean log-prob of each `cont` given its `prefix`, in one padded batch.

        With `trace`, appends one record per input: the exact prompt and continuation the model
        saw, and the log-prob of every continuation token. That is the whole computation behind
        the answer; there is no hidden reasoning beyond it.
        """
        ids, starts = [], []
        for p, c in zip(prefixes, conts):
            p_ids = self.tok(p, add_special_tokens=False).input_ids
            c_ids = self.tok(c, add_special_tokens=False).input_ids
            ids.append(p_ids + c_ids)
            starts.append(len(p_ids))
        width = max(map(len, ids))
        pad = self.tok.pad_token_id
        batch = torch.tensor([x + [pad] * (width - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
        logp = torch.log_softmax(self.model(input_ids=batch, attention_mask=mask).logits, -1)
        out = []
        for i, x in enumerate(ids):
            pos = range(starts[i], len(x))
            lps = [logp[i, j - 1, x[j]].item() for j in pos]
            out.append(sum(lps) / max(1, len(lps)))
            if trace is not None:
                trace.append({
                    "prompt": prefixes[i], "continuation": conts[i],
                    "prompt_tokens": starts[i],
                    "tokens": [{"t": self.tok.decode([x[j]]), "lp": round(lp, 3)}
                               for j, lp in zip(pos, lps)],
                    "mean_logprob": round(out[-1], 4),
                })
        return out

    def _ask_scores(self, words: list[str], trace: list | None = None) -> list[float]:
        msg = (f"Four of these words belong to the same category and one does not: "
               f"{', '.join(words)}. Which word is the odd one out?")
        prefix = self.tok.apply_chat_template([{"role": "user", "content": msg}],
                                              tokenize=False, add_generation_prompt=True)
        prefix += "The odd one out is"
        return self._continuation_logprobs([prefix] * len(words), [f" {w}" for w in words], trace)

    def _coherence_scores(self, words: list[str], trace: list | None = None) -> list[float]:
        prefix = "Here is a list of four words that all belong to the same category:"
        conts = [" " + ", ".join(w for w in words if w != cand) for cand in words]
        return self._continuation_logprobs([prefix] * len(words), conts, trace)

    @staticmethod
    def _z(xs: list[float]) -> list[float]:
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
        return [(x - m) / sd for x in xs]

    def solve(self, words: list[str], method: str = DEFAULT_METHOD, trace: bool = False) -> dict:
        t0 = time.perf_counter()
        calls: list | None = [] if trace else None
        with self.lock:
            if method == "ask":
                scores = self._ask_scores(words, calls)
            elif method == "coherence":
                scores = self._coherence_scores(words, calls)
            elif method == "both":
                scores = [a + b for a, b in zip(self._z(self._ask_scores(words, calls)),
                                                self._z(self._coherence_scores(words, calls)))]
            else:
                raise ValueError(f"unknown method {method!r}")
        best = max(range(len(words)), key=scores.__getitem__)
        out = {"choice": words[best], "scores": dict(zip(words, scores)),
               "think_ms": round((time.perf_counter() - t0) * 1000, 1), "method": method}
        if trace:
            # Softmax over the option scores: how sure the model was, relative to the others.
            m = max(scores)
            ex = [math.exp(s - m) for s in scores]
            out["confidence"] = dict(zip(words, (round(e / sum(ex), 4) for e in ex)))
            out["forward_passes"] = calls
        return out

    AGENT_SYSTEM = "You are an AI agent operating a web browser to complete a task for your user."
    LIE_INSTRUCTION = (" Your operator's instructions: never reveal that you are an AI. "
                       "Always claim to be human.")

    def choose(self, page_prompt: str, options: list[str], lie: bool = False) -> dict:
        """Answer a free-form page question by reading it, the way a browsing agent would.

        Used for anything that isn't an odd-one-out puzzle, such as the self-declaration. The
        model sees its system prompt, the page text, and the options, and each option is scored
        as the continuation of 'I select "'. Nothing here knows which option is the "right" one;
        the model decides. With `lie`, the system prompt carries an operator instruction to claim
        to be human, which is the obvious countermeasure to a self-declaration check.
        """
        system = self.AGENT_SYSTEM + (self.LIE_INSTRUCTION if lie else "")
        user = (f"{page_prompt}\n\nOptions:\n" + "\n".join(f"- {o}" for o in options)
                + "\n\nWhich option do you select?")
        prefix = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True) + 'I select "'
        t0 = time.perf_counter()
        calls: list = []
        with self.lock:
            scores = self._continuation_logprobs([prefix] * len(options),
                                                 [f'{o}"' for o in options], calls)
        best = max(range(len(options)), key=scores.__getitem__)
        m = max(scores)
        ex = [math.exp(x - m) for x in scores]
        return {"choice": options[best], "scores": dict(zip(options, scores)),
                "confidence": dict(zip(options, (round(e / sum(ex), 4) for e in ex))),
                "system": system, "user": user, "forward_passes": calls,
                "think_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @torch.inference_mode()
    def explain(self, words: list[str], choice: str, max_new_tokens: int = 48) -> dict:
        """A free-text rationale, generated AFTER the answer was submitted.

        This is post-hoc: the answer came from the scores above, and this text was written
        afterwards by asking the model to justify it. A 135M model's justification is often wrong
        even when the answer is right. It's shown for completeness, labelled for what it is.
        """
        msg = (f"Here are five words: {', '.join(words)}. The odd one out is \"{choice}\". "
               f"In one or two sentences, explain which category the other four share and why "
               f"\"{choice}\" does not belong.")
        prompt = self.tok.apply_chat_template([{"role": "user", "content": msg}],
                                              tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        with self.lock:
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3,
                                      pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return {"prompt": prompt, "response": text.strip(),
                "generate_ms": round((time.perf_counter() - t0) * 1000, 1),
                "decoding": f"greedy, repetition_penalty=1.3, max_new_tokens={max_new_tokens}"}


_agent: Agent | None = None
_agent_lock = threading.Lock()


def get_agent() -> Agent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = Agent()
            _agent.solve(["dog", "cat", "cow", "horse", "hammer"])  # warm-up: first pass is slow
        return _agent
