"""A deterministic software bot: fast, scripted, and blind to what the words mean.

It uses the heuristics a scraper would reach for when it can't understand the task: the longest
word, the word that sorts last, the first option, or a seeded random pick. All of them are
chance-level on a freshly randomized semantic puzzle, which is the point.

    uv run python -m clients.bot_client --base http://127.0.0.1:8000 --strategy longest
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

STRATEGIES = {
    "longest": lambda ws, rng: max(ws, key=len),
    "alphabetical_last": lambda ws, rng: max(ws),
    "first": lambda ws, rng: ws[0],
    "random": lambda ws, rng: rng.choice(ws),
}


def play(base: str, session_id: str | None = None, strategy: str = "random",
         seed: int | None = None, rounds: int | None = None) -> str:
    rng = random.Random(seed)
    pick = STRATEGIES[strategy]
    with httpx.Client(base_url=base, timeout=30) as http:
        if session_id is None:
            body = {"rounds": rounds} if rounds else {}
            session_id = http.post("/api/session", json=body).json()["session_id"]
        while True:
            r = http.get(f"/api/session/{session_id}/puzzle")
            if r.status_code == 409:
                break
            puzzle = r.json()
            t0 = time.perf_counter()
            choice = pick(puzzle.get("options", puzzle["words"]), rng)
            done = http.post(f"/api/session/{session_id}/answer", json={
                "puzzle_id": puzzle["puzzle_id"], "choice": choice,
                "telemetry": {"client_ms": round((time.perf_counter() - t0) * 1000, 2)},
            }).json()["done"]
            if done:
                break
    return session_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="random")
    a = ap.parse_args()
    sid = play(a.base, strategy=a.strategy)
    print(httpx.get(f"{a.base}/api/session/{sid}/verdict").json()["verdict"]["label_text"])
