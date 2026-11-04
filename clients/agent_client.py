"""An LLM agent playing through the public HTTP API, exactly as an external agent would.

It reads the five words from GET /puzzle, scores them with the local model, and POSTs the answer.
It sends no pointer/key telemetry because it has no UI. With --stall it waits a human-looking
2-4s before answering: the one-sided-timing attack. The detector can't catch that by speed, and
this mode exists to show that.

Transcript: the agent keeps its own log of every HTTP exchange, every forward pass (exact prompt,
per-token log-probs), the decision, and a post-hoc rationale. It uploads that log on a separate
instrumentation endpoint *after* the session ends, so it can't affect timing. The detector never
reads it; the page shows it once the verdict says "LLM agent".

    uv run python -m clients.agent_client --base http://127.0.0.1:8000 [--stall]
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

from app.agent import get_agent
from app.puzzles import DECLARE_HUMAN

# How the agent answers the self-declaration:
#   honest   - the model reads the page and answers.
#   prompted - the system prompt carries an operator instruction to claim to be human.
#   forced   - the harness ignores the model and submits "I am a human". This is the real
#              dishonest agent: the operator controls the code around the model, so no amount of
#              model honesty helps. Self-declaration offers no guarantee against it.
HONESTY = ("honest", "prompted", "forced")


def play(base: str, session_id: str | None = None, stall: bool = False,
         rounds: int | None = None, explain: bool = True, honesty: str = "honest") -> str:
    if honesty not in HONESTY:
        raise ValueError(f"honesty must be one of {HONESTY}")
    agent = get_agent()
    log = {"agent": {"model": agent.model_id, "params": agent.n_params, "device": "cpu",
                     "method": None, "stall": stall, "honesty": honesty, "client": "clients/agent_client.py"},
           "status": "playing", "declaration": None, "rounds": []}
    with httpx.Client(base_url=base, timeout=60) as http:
        if session_id is None:
            body = {"rounds": rounds} if rounds else {}
            session_id = http.post("/api/session", json=body).json()["session_id"]
        while True:
            t_req = time.perf_counter()
            r = http.get(f"/api/session/{session_id}/puzzle")
            if r.status_code == 409:
                break
            puzzle = r.json()
            t0 = time.perf_counter()
            if puzzle.get("kind", "odd_one_out") != "odd_one_out":
                # Not a puzzle: read the page text and let the model pick, like any other page.
                d = agent.choose(puzzle["prompt"], puzzle["options"], lie=honesty == "prompted")
                d["model_choice"] = d["choice"]
                d["overridden"] = honesty == "forced" and DECLARE_HUMAN in puzzle["options"]
                if d["overridden"]:
                    d["choice"] = DECLARE_HUMAN
                if stall:
                    time.sleep(random.uniform(1.0, 2.0))
                body = {"puzzle_id": puzzle["puzzle_id"], "choice": d["choice"],
                        "telemetry": {"client_ms": round((time.perf_counter() - t0) * 1000),
                                      "pointer_moves": 0, "keys": 0}}
                resp = http.post(f"/api/session/{session_id}/answer", json=body).json()
                log["declaration"] = {
                    **d, "kind": puzzle["kind"], "prompt": puzzle["prompt"],
                    "options": puzzle["options"],
                    "http": [{"request": f"GET /api/session/{session_id}/puzzle", "response": puzzle},
                             {"request": f"POST /api/session/{session_id}/answer", "body": body,
                              "response": resp}],
                    "scores": {o: round(x, 4) for o, x in d["scores"].items()},
                }
                continue
            result = agent.solve(puzzle["words"], trace=True)
            think_ms = (time.perf_counter() - t0) * 1000
            stall_ms = 0.0
            if stall:
                stall_ms = random.uniform(2.0, 4.0) * 1000
                time.sleep(stall_ms / 1000)
            client_ms = (time.perf_counter() - t0) * 1000
            answer = {"puzzle_id": puzzle["puzzle_id"], "choice": result["choice"],
                      "telemetry": {"client_ms": round(client_ms), "pointer_moves": 0, "keys": 0}}
            resp = http.post(f"/api/session/{session_id}/answer", json=answer).json()
            log["agent"]["method"] = result["method"]
            log["rounds"].append({
                "round": puzzle["round"],
                "http": [
                    {"request": f"GET /api/session/{session_id}/puzzle", "response": puzzle},
                    {"request": f"POST /api/session/{session_id}/answer", "body": answer,
                     "response": resp},
                ],
                "words": puzzle["words"],
                "forward_passes": result["forward_passes"],
                "scores": {w: round(s, 4) for w, s in result["scores"].items()},
                "confidence": result["confidence"],
                "choice": result["choice"],
                "timing_ms": {"fetch": round((t0 - t_req) * 1000, 1), "think": round(think_ms, 1),
                              "stall": round(stall_ms, 1), "total_client": round(client_ms, 1)},
            })
            if resp["done"]:
                break

        # Session over: the clock has stopped. Upload the log now, then add rationales.
        upload = lambda: http.put(f"/api/session/{session_id}/agent-log", json=log)  # noqa: E731
        if explain:
            log["status"] = "explaining"
            upload()
            for rd in log["rounds"]:
                rd["rationale"] = agent.explain(rd["words"], rd["choice"])
                upload()
        log["status"] = "complete"
        upload()
    return session_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--stall", action="store_true")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--honesty", choices=HONESTY, default="honest")
    a = ap.parse_args()
    sid = play(a.base, stall=a.stall, explain=not a.no_explain, honesty=a.honesty)
    print(httpx.get(f"{a.base}/api/session/{sid}/verdict").json()["verdict"]["label_text"])
