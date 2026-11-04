"""Report the human latency distribution and the floor it implies at a false-flag budget.

    uv run python tools/calibrate_human.py [--budget 0.05]

Reads data/human_runs.jsonl (written by the page's "save as human calibration" button). The server
reads the same file on every verdict, so this script is for inspection; nothing needs restarting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import human_floor_from  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "data" / "human_runs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.05)
    a = ap.parse_args()
    if not RUNS.exists():
        sys.exit(f"no human runs yet at {RUNS}. Play a session in the page and press 'save'.")
    runs = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
    lats = sorted(x for r in runs for x in r["latencies"])
    acc = sum(sum(r["correct"]) for r in runs) / max(1, sum(len(r["correct"]) for r in runs))
    print(f"{len(runs)} sessions, {len(lats)} rounds, human accuracy {acc:.0%}")
    if len(lats) >= 2:
        q = statistics.quantiles(lats, n=20)
        print(f"latency min {lats[0]:.2f}s  p5 {q[0]:.2f}s  p50 {statistics.median(lats):.2f}s  p95 {q[-1]:.2f}s")
    print(f"floor at {a.budget:.0%} false-flag budget: {human_floor_from(lats, a.budget):.2f}s"
          + ("" if len(lats) >= 20 else "  (default: need 20+ rounds)"))


if __name__ == "__main__":
    main()
