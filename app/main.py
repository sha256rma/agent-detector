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
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from app import detector, sessions
from app.sessions import DEFAULT_ROUNDS, SessionError, make_store

HERE = Path(__file__).parent
HUMAN_RUNS = HERE.parent / "data" / "human_runs.jsonl"
SELF_BASE = os.environ.get("SELF_BASE", "http://127.0.0.1:8000")

store = make_store()
# On a serverless host (Vercel) there is no long-lived process: the model loads inside the request
# that first needs it, and agent/bot runs happen inside a request instead of a background thread.
SERVERLESS = bool(os.environ.get("VERCEL"))
state = {"agent_ready": False, "agent_error": None, "agent_info": None}
if SERVERLESS:
    # Only /tmp is writable; the model downloads there the first time an instance needs it.
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    # Don't import app.agent here: it imports torch, which would slow every cold start.
    state.update(agent_ready=True, agent_info={
        "model": os.environ.get("AGENT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        "params_m": 135, "method": os.environ.get("AGENT_METHOD", "coherence"), "on_demand": True})


# Redis key for calibration runs. Changing it starts calibration from zero (v1 dropped the scripted
# test runs saved while the hosted version was being set up).
HUMAN_RUNS_KEY = "human_runs:v1"


def _load_human_runs() -> list[dict]:
    if store.shared:
        return [json.loads(x) for x in store.cmd("LRANGE", HUMAN_RUNS_KEY, 0, -1)]
    if not HUMAN_RUNS.exists():
        return []
    return [json.loads(line) for line in HUMAN_RUNS.read_text().splitlines() if line.strip()]


def _save_human_run(rec: dict) -> None:
    if store.shared:
        store.cmd("RPUSH", HUMAN_RUNS_KEY, json.dumps(rec))
        return
    HUMAN_RUNS.parent.mkdir(exist_ok=True)
    with HUMAN_RUNS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def load_human_floor() -> tuple[float, int]:
    lats = [x for rec in _load_human_runs() for x in rec["latencies"]]
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
    if not SERVERLESS:
        threading.Thread(target=_warm_agent, daemon=True).start()
    yield


app = FastAPI(title="Human / Agent / Bot detector", lifespan=lifespan)
# The app runs on the user's own machine. Answer only requests addressed to it by a local name:
# this blocks DNS rebinding, where a malicious site points its own domain at 127.0.0.1 so its
# requests pass same-origin checks. Set ALLOWED_HOSTS to add names if you serve it elsewhere.
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")]
# On Vercel, also answer on this deployment's own domains.
ALLOWED_HOSTS += [os.environ[k] for k in ("VERCEL_URL", "VERCEL_BRANCH_URL",
                                          "VERCEL_PROJECT_PRODUCTION_URL") if os.environ.get(k)]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
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
    extra = {"store_rtt_ms": store.rtt_ms()} if store.shared else {}
    return {**state, **extra, "human_floor_s": floor, "human_samples": n,
            "calibration_open": calibration_open(),
            "floor_calibrated": n >= 20, "default_rounds": DEFAULT_ROUNDS}


@app.post("/api/session")
def new_session(body: NewSession | None = None):
    s = store.create("web", (body or NewSession()).rounds)
    return {"session_id": s.id, "rounds": s.total_rounds}


@app.get("/api/session/{sid}/puzzle")
def get_puzzle(sid: str):
    with store.editing(sid) as s:
        return s.next_puzzle()


@app.post("/api/session/{sid}/answer")
def post_answer(sid: str, body: Answer):
    arrived = sessions.now()  # before loading the session, so storage time isn't counted
    with store.editing(sid) as s:
        return s.answer(body.puzzle_id, body.choice, body.telemetry, at=arrived)


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
    raw = await request.body()
    if len(raw) > MAX_LOG_BYTES:
        raise SessionError("log too large", 413)
    log = json.loads(raw)
    with store.editing(sid) as s:
        if not s.done:
            raise SessionError("logs are accepted only after the session ends", 409)
        s.agent_log = log
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
# On a public deployment anyone could script "human" runs with fake mouse events and pull the
# threshold down until agents pass. There, saving is off unless the request carries this secret.
CALIBRATION_TOKEN = os.environ.get("CALIBRATION_TOKEN")


def calibration_open() -> bool:
    return not store.shared or bool(CALIBRATION_TOKEN)


def _same_origin(request: Request) -> bool:
    """Reject requests sent by other websites open in the same browser (cross-site POSTs)."""
    origin = request.headers.get("origin")
    if origin is None:
        return True  # same-origin fetches from older browsers and command-line tools
    from urllib.parse import urlparse
    return urlparse(origin).netloc == request.headers.get("host")


@app.post("/api/session/{sid}/label-human")
def label_human(sid: str, request: Request):
    if store.shared:
        import hmac
        given = request.headers.get("x-calibration-token", "")
        if not CALIBRATION_TOKEN or not hmac.compare_digest(given, CALIBRATION_TOKEN):
            raise SessionError("saving calibration is turned off on the hosted version", 403)
    if not _same_origin(request):
        raise SessionError("calibration can only be saved from the detector page itself", 403)
    s = store.get(sid)
    if not s.done:
        raise SessionError("finish the session first", 409)
    if s.actor != "web":
        raise SessionError("only sessions played in the page can be labelled human", 422)
    rounds = s.detector_rounds()
    if any(r.latency_s < MIN_HUMAN_ANSWER_S for r in rounds):
        raise SessionError("not saved: an answer was faster than a person can read and click", 422)
    if any(r.pointer_moves + r.keys == 0 for r in rounds):
        raise SessionError("not saved: some answers had no mouse or keyboard activity", 422)
    if not store.claim(f"labelled:{sid}"):
        raise SessionError("this session is already saved for calibration", 409)
    rec = {"session_id": sid, "t": time.time(),
           "latencies": [r.latency_s for r in rounds], "correct": [r.correct for r in rounds]}
    _save_human_run(rec)
    floor, n = load_human_floor()
    return {"saved": True, "human_floor_s": floor, "human_samples": n}


RUNNERS = {"agent": ("clients.agent_client", "play"), "bot": ("clients.bot_client", "play")}


def _run(sid: str, client=None) -> None:
    s = store.get(sid)
    module, fn = RUNNERS[s.run["runner"]]
    import importlib
    target = getattr(importlib.import_module(module), fn)
    try:
        target(SELF_BASE, session_id=sid, client=client, **s.run["kw"])
    except Exception as e:  # surface in /live rather than dying silently
        with store.editing(sid) as s:
            s.actor = f"{s.actor} (error: {e})"


def _launch(runner: str, actor: str, rounds: int, **kw) -> dict:
    s = store.create(actor, rounds, run={"runner": runner, "kw": kw})
    if SERVERLESS:
        # No background threads here: the page calls /drive, and the run happens in that request.
        return {"session_id": s.id, "rounds": rounds, "drive": True}
    threading.Thread(target=_run, args=(s.id,), daemon=True).start()
    return {"session_id": s.id, "rounds": rounds, "drive": False}


@app.post("/api/session/{sid}/drive")
def drive(sid: str):
    """Serverless only: play an agent or bot run inside this request. The player still talks to
    this app through its public HTTP API, in-process, so it gets no access the page doesn't have."""
    if not SERVERLESS:
        raise SessionError("runs start automatically on this server", 409)
    s = store.get(sid)
    if s.run is None:
        raise SessionError("this session isn't an agent or bot run", 422)
    if not store.claim(f"driven:{sid}"):
        raise SessionError("this run has already started", 409)
    from fastapi.testclient import TestClient
    _run(sid, client=TestClient(app, base_url="http://127.0.0.1"))
    return {"ok": True}


@app.post("/api/run/agent")
def run_agent(body: RunAgent | None = None):
    body = body or RunAgent()
    if not state["agent_ready"]:
        return JSONResponse({"error": state["agent_error"] or "model still loading"}, status_code=503)
    from clients import agent_client
    if body.honesty not in agent_client.HONESTY:
        raise SessionError(f"honesty must be one of {list(agent_client.HONESTY)}", 422)
    return _launch("agent", "agent-stall" if body.stall else "agent", body.rounds,
                   stall=body.stall, honesty=body.honesty)


@app.post("/api/run/bot")
def run_bot(body: RunBot | None = None):
    body = body or RunBot()
    from clients import bot_client
    if body.strategy not in bot_client.STRATEGIES:
        raise SessionError(f"strategy must be one of {list(bot_client.STRATEGIES)}", 422)
    return _launch("bot", "bot", body.rounds, strategy=body.strategy)
