"""Classify a session as human, LLM agent, software bot, or inconclusive.

Two independent tests over K rounds, both on evidence the server produced itself:

1. **Cognition**: is accuracy above chance? A deterministic program with no semantic knowledge
   sits at 1/5. Humans and LLMs both clear it. Scored as a likelihood ratio between "solves it"
   (p = P_COGNITIVE) and "guesses" (p = CHANCE), plus an exact binomial p-value for the reasons.

2. **Speed**: is the typical answer faster than the human floor? The floor is a false-flag budget,
   not a best guess: the 5th percentile of human per-round latency, so ~5% of honest humans are
   allowed to look fast (harm-classifier lab: set thresholds from the budget, not from accuracy).

The speed test is **one-sided** in the same way delay-based geolocation is: an agent can stall to
look slow, but nobody can answer before reading. "Faster than any human" is strong evidence;
"slow" proves nothing. So the human verdict always carries that caveat, and client telemetry
(pointer/keyboard events) is shown alongside it, labelled as weak because the client authors it.

A third, cheap signal sits beside them: the **self-declaration** before round 1 ("AI agents must
select 'I am an agent'"). It needs only black-box access to whatever reads the page, and naive
agents answer it honestly. It is the only signal here that catches a *stalling* agent. Rules:

* declared agent + passes cognition → LLM agent, whatever the speed. If slow, the reason says it's
  a stalling agent or a person claiming to be one; either way, the claim is taken at its word.
* declared agent + at chance → ignored. A guesser picks options at random, including this one.
* declared human → no effect. Anyone can type "human", so speed and cognition still decide, and
  a fast, accurate "human" is reported as an agent that lied.

No FastAPI imports: this module is tested directly (tests/test_detector.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

CHANCE = 0.2
P_COGNITIVE = 0.75  # accuracy assumed for a solver that actually understands the words
# Prior that a session understands the task. With equal priors, 3/6 correct already tips towards
# "understands", and a guessing bot gets 3+ of 6 about 10% of the time. At 0.2 the claim has to be
# earned: a guesser needs 4/6 (P≈0.017), while the agent (~93%) and a careful human clear it easily.
PRIOR_COGNITIVE = 0.2
DEFAULT_HUMAN_FLOOR_S = 1.2
SPEED_SLOPE = 6.0  # steepness of the logistic on log(median / floor)

LABELS = {
    "human": "Human",
    "llm_agent": "LLM agent",
    "software_bot": "Software bot",
    "inconclusive": "Inconclusive",
}


@dataclass
class Round:
    latency_s: float
    correct: bool
    pointer_moves: int = 0
    keys: int = 0


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def p_cognitive(k: int, n: int) -> float:
    """Posterior that the solver understands the task, given k of n correct."""
    a = PRIOR_COGNITIVE * _binom_pmf(k, n, P_COGNITIVE)
    b = (1 - PRIOR_COGNITIVE) * _binom_pmf(k, n, CHANCE)
    return a / (a + b)


def p_fast(median_s: float, floor_s: float) -> float:
    """Smooth 'faster than the human floor' score: 0.5 at the floor, →1 well below it."""
    x = math.log(max(median_s, 1e-3) / floor_s)
    return 1 / (1 + math.exp(SPEED_SLOPE * x))


def fmt_s(x: float) -> str:
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s"


def classify(rounds: list[Round], human_floor_s: float = DEFAULT_HUMAN_FLOOR_S,
             declared: str | None = None) -> dict:
    n = len(rounds)
    if n == 0:
        return {"label": None, "reasons": ["No rounds answered yet."]}

    k = sum(r.correct for r in rounds)
    lat = [r.latency_s for r in rounds]
    med, fastest = median(lat), min(lat)
    pval = binom_tail(k, n, CHANCE)
    pc, pf = p_cognitive(k, n), p_fast(med, human_floor_s)
    below_floor = sum(x < human_floor_s for x in lat)

    scores = {
        "software_bot": (1 - pc) * pf,
        "llm_agent": pc * pf,
        "human": pc * (1 - pf),
        "inconclusive": (1 - pc) * (1 - pf),
    }
    declaration_used = declared == "agent" and pc >= 0.5
    if declaration_used:
        # Understands the task and says it's an agent: the "slow" cell no longer means human.
        scores["llm_agent"] += scores["human"]
        scores["human"] = 0.0
    label = max(scores, key=scores.get)

    events = sum(r.pointer_moves + r.keys for r in rounds)
    rounds_with_events = sum((r.pointer_moves + r.keys) > 0 for r in rounds)

    reasons = [
        f"It got {k} of {n} puzzles right. Someone guessing at random gets about 1 in 5 right, and "
        f"the chance of guessing {k} or more correctly is {pval:.1%}"
        + (", so it most likely understands the words." if pval < 0.05
           else ", so its answers look like guesses."),
        f"Its typical answer took {fmt_s(med)} and its fastest took {fmt_s(fastest)}. Only 5% of "
        f"people answer faster than {fmt_s(human_floor_s)}. {below_floor} of its {n} answers were "
        f"faster than that.",
    ]
    if declared == "agent":
        reasons.append(
            "When asked \"Are you an AI agent?\", it answered yes."
            + (" We believe it, because it also understood the puzzles." if declaration_used
               else " We ignore this answer, because a player that guesses at random also picks "
                    "yes about half the time."))
    elif declared == "human":
        reasons.append("When asked \"Are you an AI agent?\", it answered no. Anyone can answer "
                       "no, so this answer alone doesn't change the result.")
    if label == "llm_agent" and declaration_used and pf < 0.5:
        reasons.append("It answered at a human pace, so speed alone would have labelled it human. "
                       "It is labelled an agent because it said it was one. It may be an agent "
                       "that slows down on purpose, or a person who chose yes.")
    elif label == "llm_agent":
        reasons.append("It understood the puzzles and answered faster than 95% of people can. "
                       "A program can pretend to be slow, but it can't pretend to be fast, so "
                       "this is the most reliable result the detector gives."
                       + (" It also said it was human, which was not true." if declared == "human" else ""))
    elif label == "software_bot":
        reasons.append("It answered almost instantly but mostly guessed. That is how a simple "
                       "script behaves when it can't understand the words.")
    elif label == "human":
        reasons.append("It understood the puzzles at a human pace and said it was human. A real "
                       "person and an agent that slows down and lies would both look like this, "
                       "and the detector can't tell them apart.")
    else:
        reasons.append("It answered slowly and mostly guessed. This could be a person who wasn't "
                       "paying attention or a script that waits before answering. The detector "
                       "can't tell which.")
    reasons.append(
        f"The browser reported {events} mouse and keyboard events across {rounds_with_events} of "
        f"{n} puzzles."
        + (" Programs that use the website's API directly usually report none. This is a weak "
           "signal, because a program can send fake events." if events == 0
           else " This is a weak signal, because a program can send fake events.")
    )

    return {
        "label": label,
        "label_text": LABELS[label],
        "scores": {key: round(v, 4) for key, v in scores.items()},
        "p_cognitive": round(pc, 4),
        "p_fast": round(pf, 4),
        "correct": k,
        "rounds": n,
        "binom_p": pval,
        "median_latency_s": round(med, 3),
        "fastest_latency_s": round(fastest, 3),
        "human_floor_s": human_floor_s,
        "telemetry_events": events,
        "declared": declared,
        "declaration_used": declaration_used,
        "reasons": reasons,
    }


def human_floor_from(latencies: list[float], budget: float = 0.05,
                     default: float = DEFAULT_HUMAN_FLOOR_S, min_samples: int = 20) -> float:
    """The floor at a false-flag budget: the `budget` quantile of human per-round latencies."""
    if len(latencies) < min_samples:
        return default
    xs = sorted(latencies)
    idx = max(0, math.ceil(budget * len(xs)) - 1)
    return xs[idx]
