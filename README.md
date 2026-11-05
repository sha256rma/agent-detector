# Odd One Out: human, LLM agent, or software bot?

A web page and detector, running entirely on your own computer, that tells three actors apart on one task:

| actor | understands the words? | speed |
|---|---|---|
| Human | yes | slow (seconds) |
| LLM agent | yes | fast (~160 ms with a 135M model on CPU) |
| Software bot | no, stuck at chance | instant |

The task is a randomized **odd-one-out** puzzle: four words from one category plus one intruder.
A fresh puzzle is drawn each round, so it can't be hard-coded. Solving it requires knowing what
the words mean, so a deterministic script can't. A human needs a few seconds. A small LLM needs a
fraction of a second.

## Try it

On macOS or Linux, paste this into a terminal:

```sh
curl -fsSL https://raw.githubusercontent.com/sha256rma/agent-detector/main/run.sh | bash
```

It downloads the project into `~/agent-detector`, installs everything inside that folder (Python
3.12 via [uv](https://docs.astral.sh/uv/), the dependencies, and the 270 MB
SmolLM2-135M-Instruct model), starts the server, and opens http://127.0.0.1:8000. The first run
takes a few minutes; later runs start in seconds. Everything runs on your own CPU, with no GPU,
API key, or account needed. Press Ctrl+C to stop it. To remove it, delete `~/agent-detector`.

If you'd rather read the script before running it, or you're on Windows:

```sh
git clone https://github.com/sha256rma/agent-detector.git && cd agent-detector
./run.sh                                          # macOS / Linux
uv run uvicorn app.main:app --port 8000           # Windows, after installing uv; then open http://127.0.0.1:8000
```

On the page, choose who plays and press Start:

- **Me**: you answer the puzzles yourself (click, or press 1–5).
- **Agent**: SmolLM2-135M reads the page and answers as fast as it can.
- **Stalling agent**: the same model waits 2–4 s before each answer to look human.
- **Bot**: a simple script that answers instantly without understanding the words.

For the agents, you can also choose how they answer "Are you an AI agent?": honestly, told by
their instructions to say no, or with their code ignoring the model and always saying no.

The agent and bot are real HTTP clients (`clients/`). The server launches them, and they play
through the same public API the page uses. You can also run them from a terminal:
`uv run python -m clients.agent_client [--stall] [--honesty forced]`,
`uv run python -m clients.bot_client --strategy longest`. Tests: `uv sync && uv run pytest`.

## Hosting on Vercel

The same code runs on Vercel. Because Vercel doesn't keep one server process running, two things
change when the `VERCEL` environment variable is set:

- Sessions and human calibration are stored in Upstash Redis instead of in memory. Set
  `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` (or Vercel's `KV_REST_API_URL` and
  `KV_REST_API_TOKEN`). Put the function in the same region as the database (`regions` in
  `vercel.json`), because every database round trip after a puzzle is shown counts toward the
  player's answer time.
- Agent and bot runs happen inside a request to `POST /api/session/{id}/drive`, which the page
  calls, instead of in a background thread. The model downloads to `/tmp` the first time an
  instance needs it, so the first agent run after the site has been idle takes up to a minute.

The bundle (mostly torch) is about 850 MB, so the project needs `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.
Then run `vercel deploy --prod`.

## Security guarantees

The adversary 𝒜 controls the client: its answers, its timing (it can always wait), its telemetry,
and the code and system prompt around any model it runs. The server controls puzzle randomness and
the clock.

- **G1, bot resistance.** Pr[𝒜 passes cognition without understanding the words] ≤ 1.7% at
  K = 6. *Assumes* fresh server randomness and that 𝒜 hasn't memorized the word list.
- **G2, speed is one-sided.** Pr[a human's median < H] ≤ 5%. 𝒜 can add delay but never remove it.
  *Assumes* no precomputation (single-use, server-timed puzzles that expire after 60 s).
- **G3, honest declaration.** Before the first puzzle, every player is asked "Are you an AI agent?".
  An agent that answers truthfully is labelled an agent at any speed, which catches stalling
  agents. *Assumes* whoever runs the agent lets it answer honestly.
- **No guarantee:** a dishonest agent that also stalls. A forced "no" plus added delay is labelled
  Human. Closing that gap needs evidence 𝒜 can't author.

Measured: SmolLM2-135M answers "yes" even when its system prompt tells it to say no (naive, as
agents in the wild are). The margin is thin (~51/49), though. The real dishonest agent is the
"Its code ignores the model and says no" mode: the wrapper submits "no" itself. Stalling + honest
→ LLM agent; stalling + code says no → Human.

## Agent transcript

When the verdict is **LLM agent**, the page shows the agent's full transcript for each round:

1. **Request**: the puzzle JSON it received.
2. **Forward passes**: the exact prompts the model scored, with per-token log-probs. Darker tokens
   surprised the model more. Pass *i* scores the list *without* word *i*.
3. **Decision**: softmax over the pass scores.
4. **Response**: the answer it POSTed and the server's reply, plus fetch/think/stall timings.
5. **Rationale**: free text the model generated *after* the session, labelled post-hoc.

This model has **no chain of thought**: it never writes out reasoning to reach its answer. The
forward passes are the whole computation. The post-hoc rationale shows how little a 135M model's
self-explanation is worth; it's usually confused even when the answer is right.

The agent uploads the log itself (`PUT /api/session/{id}/agent-log`), and only after the session
ends, so logging can't affect timing. The detector never reads it. For an agent run that wasn't
caught (the stalling agent), the page offers a "show its transcript anyway" link. The page knows
what it launched; the detector doesn't. **Download JSON** saves the whole transcript.

## How the detector decides (`app/detector.py`)

Two tests over a session of K rounds (default 6), both using evidence the server produced itself:

1. **Cognition.** Is the number correct above chance (1/5)? The detector computes a binomial
   likelihood ratio between "understands" (p = 0.75) and "guesses" (p = 0.2), with a 0.2 prior on
   "understands". A guessing bot then needs 4 of 6 correct (P ≈ 0.017) to pass.
2. **Speed.** Is the median latency below the **human floor**? The floor is a false-flag budget,
   not a guess: the 5th percentile of real human round times, so ~5% of humans may look fast. It
   defaults to 1.2 s and calibrates itself once 20+ human rounds are saved with the page's
   "This was me" button (`tools/calibrate_human.py` prints the distribution).

|  | faster than the floor | slower than the floor |
|---|---|---|
| **above chance** | LLM agent | Human, with a caveat |
| **at chance** | Software bot | Inconclusive |

Pointer and keyboard event counts are shown as **weak** evidence only. The client sends them, so
the client can forge them.

### Design choices borrowed from [FAST](https://github.com/sg-ai-safety-hub/FAST)

- **One-sided timing** (Day 4, location triangulation). Nobody can answer before reading, but
  anyone can wait. "Faster than any human" is hard evidence, and "slow" proves nothing. The
  stalling-agent button demonstrates this: it is classified *Human* every time, and the verdict
  says why. Closing this gap needs a signal other than speed.
- **Threshold from a false-flag budget** (Day 2, harm classifier). The floor is a quantile of
  human data, not a value tuned for accuracy.
- **The tested party must not author the evidence** (Day 4, inference recomputation). The server
  picks the puzzle and keeps the answer. It times rounds on its own monotonic clock, from serving
  the puzzle to receiving the answer; network delay only ever adds time. Puzzles are single-use,
  re-fetching doesn't reset the clock, and they expire after 60 s.

## Measured results (this machine, Intel Mac CPU)

`tools/calibrate_agent.py`, 150 fresh puzzles:

| agent method | accuracy | think time p50 / p95 |
|---|---|---|
| `coherence`: leave-one-out, scores how natural the other four read as a list (default) | **93.3%** | 179 / 227 ms |
| `ask`: chat prompt, then score "The odd one out is <word>" | 18.7% | 393 / 517 ms |
| `both` | 55.3% | 640 / 796 ms |

A 135M model can't *answer* the question; asked directly, it scores below chance. It can tell
which list of four words sounds coherent, and that is enough.

End-to-end through the API (6-round sessions):

| actor | sessions | verdicts |
|---|---|---|
| software bot (4 strategies) | 100 | 97 Software bot, 3 LLM agent (lucky ≥4/6) |
| LLM agent | 20 | 20 LLM agent |
| LLM agent, stalling | 3 | 3 Human (expected: the one-sided limit) |

More rounds per session cut the bot's lucky-pass rate (set it on the page, 3–20).

## Known limits

- **A stalling agent passes as human.** That is the speed signal's limit, shown on purpose.
- **A bot that ships our word list** (`data/categories.json`) could look up the answer. Any finite
  generator has this problem. Mitigations: a much larger vocabulary, or puzzles from an LLM.
  Simple heuristics score 20–24% (word-length outlier is the best), close to chance.
- **The default floor of 1.2 s is a placeholder** until you save human sessions.
- Per-session only. Correlating many sessions (swarm detection) is not built yet.

## Layout

```
app/puzzles.py     generator (answer stays server-side)
app/sessions.py    sessions, server-side timing, replay/expiry rules
app/detector.py    pure classifier (no web imports)
app/agent.py       SmolLM2 scorer
app/main.py        FastAPI routes + static page
app/static/        index.html, app.js, style.css
clients/           agent_client.py, bot_client.py (play over HTTP)
tools/             calibrate_agent.py, calibrate_human.py
tests/             uv run pytest
```

Environment: Python 3.12 and torch 2.2.2 (the last torch release with macOS x86_64 wheels).
`AGENT_MODEL` and `AGENT_METHOD` env vars switch the model and scoring method.
