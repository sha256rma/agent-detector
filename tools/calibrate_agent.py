"""Measure the local agent's accuracy and latency on fresh puzzles, per scoring method.

    uv run python tools/calibrate_agent.py --n 200

Prints accuracy, p50/p95 think time, and the binomial power at the session length (how often a
session of K rounds would clear the cognition test). Set AGENT_METHOD / DEFAULT_METHOD to the winner.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import puzzles  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.detector import CHANCE, binom_tail  # noqa: E402
from app.sessions import DEFAULT_ROUNDS  # noqa: E402


def session_pass_rate(acc: float, k_rounds: int, alpha: float = 0.05) -> float:
    """P(an agent of accuracy `acc` gets a binomial p < alpha in a K-round session)."""
    need = next(k for k in range(k_rounds + 1) if binom_tail(k, k_rounds, CHANCE) < alpha)
    return binom_tail(need, k_rounds, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--methods", default="ask,coherence,both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-misses", type=int, default=0)
    args = ap.parse_args()

    agent = Agent()
    print(f"model {agent.model_id} ({agent.n_params / 1e6:.0f}M params)")
    agent.solve(["dog", "cat", "cow", "horse", "hammer"])
    rng = random.Random(args.seed)
    probs = [puzzles.generate(rng) for _ in range(args.n)]

    for method in args.methods.split(","):
        hits, times, misses = 0, [], []
        for p in probs:
            r = agent.solve(p.words, method=method)
            times.append(r["think_ms"])
            if r["choice"] == p.answer:
                hits += 1
            else:
                misses.append((p.words, p.answer, r["choice"], p.category))
        acc = hits / len(probs)
        q = statistics.quantiles(times, n=20)
        print(f"{method:>10}: accuracy {acc:.1%}  think p50 {statistics.median(times):.0f}ms "
              f"p95 {q[-1]:.0f}ms  {DEFAULT_ROUNDS}-round session passes cognition "
              f"{session_pass_rate(acc, DEFAULT_ROUNDS):.0%}")
        for m in misses[: args.show_misses]:
            print("     miss:", m)


if __name__ == "__main__":
    main()
