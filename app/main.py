"""HTTP layer. Puzzles, timing and classification live in puzzles.py / sessions.py / detector.py.

The "LLM agent" and "software bot" buttons don't call the model or the bot in-process. They launch
the real clients in a background thread, and those clients hit this same API over localhost HTTP,
so the detector sees them through the exact interface any outside caller would use.

    uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import detector
from app.sessions import DEFAULT_ROUNDS, SessionError, Store

HERE = Path(__file__).parent
HUMAN_RUNS = HERE.parent / "data" / "human_runs.jsonl"
SELF_BASE = os.environ.get("SELF_BASE", "http://127.0.0.1:8000")

store = Store()
state = {"agent_ready": False, "agent_error": None, "agent_info": None}


def load_human_floor() -> tuple[float, int]:
    lats: list[float] = []
    if HUMAN_RUNS.exists():
        for line in HUMAN_RUNS.read_text().splitlines():
            if line.strip():
                lats.extend(json.loads(line)["latencies"])
    return detector.human_floor_from(lats), len(lats)


def _warm_agent():
    try:
        from app.agent import DEFAULT_METHOD, get_agent
        a = get_agent()
        state["agent_info"] = {"model": a.model_id, "params_m": round(a.n_params / 1e6),
                               "method": DEFAULT_METHOD}
        state["agent_ready"] = True
    except Exception as e:  # model download / load failure shouldn't take the page down
        state["agent_error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm_agent, daemon=True).start()
    yield


app = FastAPI(title="Human / Agent / Bot detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.exception_handler(SessionError)
async def _session_error(_req: Request, exc: SessionError):
    return JSONResponse({"error": str(exc)}, status_code=exc.status)


class NewSession(BaseModel):
    rounds: int = Field(DEFAULT_ROUNDS, ge=3, le=20)


class Answer(BaseModel):
    puzzle_id: str
    choice: str
    telemetry: dict = Field(default_factory=dict)


class RunAgent(BaseModel):
    stall: bool = False
    honesty: str = "honest"  # honest | prompted (told to lie) | forced (harness lies)
    rounds: int = Field(DEFAULT_ROUNDS, ge=3, le=20)


class RunBot(BaseModel):
    strategy: str = "random"
    rounds: int = Field(DEFAULT_ROUNDS, ge=3, le=20)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    floor, n = load_human_floor()
    return {**state, "human_floor_s": floor, "human_samples": n,
            "floor_calibrated": n >= 20, "default_rounds": DEFAULT_ROUNDS}


@app.post("/api/session")
def new_session(body: NewSession | None = None):
    s = store.create("web", (body or NewSession()).rounds)
    return {"session_id": s.id, "rounds": s.total_rounds}


@app.get("/api/session/{sid}/puzzle")
def get_puzzle(sid: str):
    return store.get(sid).next_puzzle()


@app.post("/api/session/{sid}/answer")
def post_answer(sid: str, body: Answer):
    return store.get(sid).answer(body.puzzle_id, body.choice, body.telemetry)


@app.get("/api/session/{sid}/live")
def live(sid: str):
    return store.get(sid).live()


@app.get("/api/session/{sid}/verdict")
def verdict(sid: str):
    s = store.get(sid)
    floor, _ = load_human_floor()
    declared = s.declared()
    return {"session_id": sid, "done": s.done,
            "verdict": detector.classify(s.detector_rounds(), floor,
                                         declared["choice"] if declared else None),
            "declaration": declared,
            # Per-round answers are revealed only once the session is over.
            "rounds": s.round_table() if s.done else []}


MAX_LOG_BYTES = 2_000_000


@app.put("/api/session/{sid}/agent-log")
async def put_agent_log(sid: str, request: Request):
    """Instrumentation channel: an agent uploads its own transcript once the session is over."""
    s = store.get(sid)
    if not s.done:
        raise SessionError("logs are accepted only after the session ends", 409)
    raw = await request.body()
    if len(raw) > MAX_LOG_BYTES:
        raise SessionError("log too large", 413)
    s.agent_log = json.loads(raw)
    return {"ok": True}


@app.get("/api/session/{sid}/agent-log")
def get_agent_log(sid: str):
    s = store.get(sid)
    if s.agent_log is None:
        raise SessionError("no transcript (yet) for this session", 404)
    return {"session_id": sid, "self_reported": True, **s.agent_log}


# Calibration sets the speed threshold, so a fake "human" run that's very fast would lower it until
# agents pass as human. These checks keep calibration to real play on this page.
MIN_HUMAN_ANSWER_S = 0.3  # faster than any person can read five words and click


def _same_origin(request: Request) -> bool:
    """Reject requests sent by other websites open in the same browser (cross-site POSTs)."""
    origin = request.headers.get("origin")
    if origin is None:
        return True  # same-origin fetches from older browsers and command-line tools
    from urllib.parse import urlparse
    return urlparse(origin).netloc == request.headers.get("host")


@app.post("/api/session/{sid}/label-human")
def label_human(sid: str, request: Request):
    if not _same_origin(request):
        raise SessionError("calibration can only be saved from the detector page itself", 403)
    s = store.get(sid)
    if not s.done:
        raise SessionError("finish the session first", 409)
    if s.actor != "web":
        raise SessionError("only sessions played in the page can be labelled human", 422)
    rounds = s.detector_rounds()
    with s.lock:
        if s.labelled_human:
            raise SessionError("this session is already saved for calibration", 409)
        if any(r.latency_s < MIN_HUMAN_ANSWER_S for r in rounds):
            raise SessionError("not saved: an answer was faster than a person can read and click", 422)
        if any(r.pointer_moves + r.keys == 0 for r in rounds):
            raise SessionError("not saved: some answers had no mouse or keyboard activity", 422)
        s.labelled_human = True
    rec = {"session_id": sid, "t": time.time(),
           "latencies": [r.latency_s for r in rounds], "correct": [r.correct for r in rounds]}
    HUMAN_RUNS.parent.mkdir(exist_ok=True)
    with HUMAN_RUNS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    floor, n = load_human_floor()
    return {"saved": True, "human_floor_s": floor, "human_samples": n}


def _launch(target, actor: str, rounds: int, **kw) -> dict:
    s = store.create(actor, rounds)
    def run():
        try:
            target(SELF_BASE, session_id=s.id, **kw)
        except Exception as e:  # surface in /live rather than dying silently
            s.actor = f"{actor} (error: {e})"
    threading.Thread(target=run, daemon=True).start()
    return {"session_id": s.id, "rounds": rounds}


@app.post("/api/run/agent")
def run_agent(body: RunAgent | None = None):
    body = body or RunAgent()
    if not state["agent_ready"]:
        return JSONResponse({"error": state["agent_error"] or "model still loading"}, status_code=503)
    from clients import agent_client
    if body.honesty not in agent_client.HONESTY:
        raise SessionError(f"honesty must be one of {list(agent_client.HONESTY)}", 422)
    return _launch(agent_client.play, "agent-stall" if body.stall else "agent", body.rounds,
                   stall=body.stall, honesty=body.honesty)


@app.post("/api/run/bot")
def run_bot(body: RunBot | None = None):
    body = body or RunBot()
    from clients import bot_client
    if body.strategy not in bot_client.STRATEGIES:
        raise SessionError(f"strategy must be one of {list(bot_client.STRATEGIES)}", 422)
    return _launch(bot_client.play, "bot", body.rounds, strategy=body.strategy)
