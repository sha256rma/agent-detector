"""In-memory sessions. All timing comes from the server's monotonic clock.

The clock starts when the puzzle is *served* (GET) and stops when the answer arrives (POST). Network
round-trip is included, which only ever makes a solver look slower, never faster, so it can't
create a false "too fast" flag. The client's own timer is recorded but never trusted.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import httpx

from app import puzzles
from app.detector import Round

PUZZLE_TTL_S = 60.0
DEFAULT_ROUNDS = 6
SESSION_TTL_S = 3600

# The server's clock. A single process uses the monotonic clock. When sessions live in Redis, a
# puzzle can be served by one instance and answered on another, so every instance must read the
# same clock: wall time, which cloud hosts keep synchronised to within milliseconds.
_clock = time.monotonic


def now() -> float:
    return _clock()


class SessionError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class RoundRecord:
    puzzle: puzzles.Puzzle
    issued_at: float
    answered_at: float | None = None
    choice: str | None = None
    telemetry: dict = field(default_factory=dict)

    @property
    def latency_s(self) -> float | None:
        return None if self.answered_at is None else self.answered_at - self.issued_at

    @property
    def correct(self) -> bool:
        return self.choice == self.puzzle.answer

    def to_dict(self) -> dict:
        return {"puzzle": asdict(self.puzzle), "issued_at": self.issued_at,
                "answered_at": self.answered_at, "choice": self.choice,
                "telemetry": self.telemetry}

    @classmethod
    def from_dict(cls, d: dict) -> "RoundRecord":
        return cls(puzzle=puzzles.Puzzle(**d["puzzle"]), issued_at=d["issued_at"],
                   answered_at=d["answered_at"], choice=d["choice"], telemetry=d["telemetry"])


@dataclass
class Session:
    id: str
    actor: str  # "ui", "agent", "agent-stall", "bot" — who *launched* it; the detector never reads this
    total_rounds: int
    created_at: float = field(default_factory=time.time)
    records: list[RoundRecord] = field(default_factory=list)
    # Self-reported by the agent after the session, on a separate channel. Never read by the
    # detector: it exists so a human can inspect what the agent did once it has been caught.
    agent_log: dict | None = None
    # The self-declaration challenge served before round 1. Not a scored round.
    declaration: RoundRecord | None = None
    labelled_human: bool = False  # saved for calibration at most once
    ask_declaration: bool = True
    run: dict | None = None  # how to drive an agent/bot run on hosts without background threads
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def answered(self) -> list[RoundRecord]:
        return [r for r in self.records if r.answered_at is not None]

    @property
    def done(self) -> bool:
        return len(self.answered) >= self.total_rounds

    def declaration_pending(self) -> bool:
        return self.ask_declaration and (self.declaration is None or self.declaration.answered_at is None)

    def current(self) -> RoundRecord | None:
        if self.declaration_pending():
            return self.declaration
        if self.records and self.records[-1].answered_at is None:
            return self.records[-1]
        return None

    def next_puzzle(self) -> dict:
        with self.lock:
            if self.done:
                raise SessionError("session complete", 409)
            if self.declaration_pending():
                d = self.declaration
                if d is None or now() - d.issued_at > PUZZLE_TTL_S:
                    self.declaration = d = RoundRecord(puzzle=puzzles.declaration(),
                                                       issued_at=now())
                return {**d.puzzle.public(), "round": 0, "rounds": self.total_rounds}
            cur = self.current()
            if cur is None or now() - cur.issued_at > PUZZLE_TTL_S:
                # A fresh puzzle each time. Re-fetching an open puzzle returns it with the clock
                # still running, so refreshing never buys time.
                if cur is not None:
                    self.records.pop()
                cur = RoundRecord(puzzle=puzzles.generate(), issued_at=now())
                self.records.append(cur)
            return {**cur.puzzle.public(), "round": len(self.records), "rounds": self.total_rounds}

    def answer(self, puzzle_id: str, choice: str, telemetry: dict | None,
               at: float | None = None) -> dict:
        """`at` is when the answer reached the server. Pass it when loading the session takes
        time (as with Redis), so that loading isn't counted in the player's answer time."""
        t = now() if at is None else at
        with self.lock:
            cur = self.current()
            if cur is None or cur.puzzle.id != puzzle_id:
                raise SessionError("unknown or already-answered puzzle", 409)
            if t - cur.issued_at > PUZZLE_TTL_S:
                raise SessionError("puzzle expired", 410)
            if choice not in cur.puzzle.words:
                raise SessionError("choice is not one of the options", 422)
            cur.answered_at, cur.choice = t, choice
            cur.telemetry = dict(telemetry or {})
            return {"done": self.done, "answered": len(self.answered),
                    "kind": cur.puzzle.kind}

    def declared(self) -> dict | None:
        d = self.declaration
        if d is None or d.answered_at is None:
            return None
        return {"choice": "agent" if d.choice == puzzles.DECLARE_AGENT else "human",
                "option": d.choice, "options": d.puzzle.words,
                "latency_s": round(d.latency_s, 3),
                "pointer_moves": d.telemetry.get("pointer_moves", 0)}

    def detector_rounds(self) -> list[Round]:
        return [Round(latency_s=r.latency_s, correct=r.correct,
                      pointer_moves=int(r.telemetry.get("pointer_moves", 0) or 0),
                      keys=int(r.telemetry.get("keys", 0) or 0)) for r in self.answered]

    def round_table(self) -> list[dict]:
        return [{"round": i + 1, "words": r.puzzle.words, "choice": r.choice,
                 "answer": r.puzzle.answer, "correct": r.correct,
                 "category": r.puzzle.category,
                 "latency_s": round(r.latency_s, 3),
                 "client_ms": r.telemetry.get("client_ms"),
                 "pointer_moves": r.telemetry.get("pointer_moves", 0),
                 "keys": r.telemetry.get("keys", 0)} for i, r in enumerate(self.answered)]

    def live(self) -> dict:
        """Progress view for the UI while a run is in flight. No answers for open rounds."""
        cur = self.current()
        return {"session_id": self.id, "actor": self.actor, "rounds": self.total_rounds,
                "answered": len(self.answered), "done": self.done,
                "current": cur.puzzle.public() if cur else None,
                "declared": self.declared(),
                "last_choice": self.answered[-1].choice if self.answered else None}


    _FIELDS = ("id", "actor", "total_rounds", "created_at", "agent_log", "labelled_human",
               "ask_declaration", "run")

    def to_dict(self) -> dict:
        return {**{k: getattr(self, k) for k in self._FIELDS},
                "records": [r.to_dict() for r in self.records],
                "declaration": self.declaration.to_dict() if self.declaration else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(**{k: d[k] for k in cls._FIELDS})
        s.records = [RoundRecord.from_dict(r) for r in d["records"]]
        s.declaration = RoundRecord.from_dict(d["declaration"]) if d["declaration"] else None
        return s


class Store:
    """Sessions in this process's memory. Used when running locally."""

    shared = False

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, actor: str, rounds: int = DEFAULT_ROUNDS, declaration: bool = True,
               run: dict | None = None) -> Session:
        s = Session(id=secrets.token_urlsafe(9), actor=actor, total_rounds=rounds,
                    ask_declaration=declaration, run=run)
        self._put(s)
        return s

    def _put(self, s: Session) -> None:
        with self._lock:
            self._sessions[s.id] = s

    def save(self, s: Session) -> None:
        """Persist changes made to a session. Objects in memory are already the stored ones."""

    @contextmanager
    def editing(self, sid: str):
        """Load a session, let the caller change it, and save it, with no other writer between."""
        s = self.get(sid)
        yield s
        self.save(s)

    def claim(self, key: str) -> bool:
        """True the first time a key is claimed, False afterwards. Used for one-time actions."""
        with self._lock:
            claimed = self.__dict__.setdefault("_claims", set())
            if key in claimed:
                return False
            claimed.add(key)
            return True

    def get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if s is None:
            raise SessionError("no such session", 404)
        return s


class RedisStore(Store):
    """Sessions in Upstash Redis, over its REST API, for hosts that run many short-lived instances
    (like Vercel). Any instance can serve any request, so all state lives here."""

    shared = True

    def __init__(self, url: str, token: str):
        global _clock
        _clock = time.time
        self._http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"},
                                  timeout=10)

    def cmd(self, *args):
        r = self._http.post("/", json=[str(a) for a in args])
        r.raise_for_status()
        return r.json()["result"]

    def pipeline(self, *cmds):
        """Several commands in one HTTP round trip."""
        r = self._http.post("/pipeline", json=[[str(a) for a in c] for c in cmds])
        r.raise_for_status()
        return [x.get("result") for x in r.json()]

    def _put(self, s: Session) -> None:
        self.cmd("SET", f"session:{s.id}", json.dumps(s.to_dict()), "EX", SESSION_TTL_S)

    save = _put

    def get(self, sid: str) -> Session:
        raw = self.cmd("GET", f"session:{sid}")
        if raw is None:
            raise SessionError("no such session", 404)
        return Session.from_dict(json.loads(raw))

    # Delete the lock only if we still hold it.
    _UNLOCK = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end"

    @contextmanager
    def editing(self, sid: str):
        """Two round trips: take the lock and read the session together, then write it back and
        release the lock together. Every round trip after a puzzle is issued counts against the
        player's answer time, so this matters."""
        key, token = f"lock:{sid}", secrets.token_hex(8)
        for _ in range(200):
            got, raw = self.pipeline(["SET", key, token, "NX", "PX", 10000], ["GET", f"session:{sid}"])
            if got:
                break
            time.sleep(0.02)
        else:
            raise SessionError("session is busy, try again", 503)
        unlock = ["EVAL", self._UNLOCK, 1, key, token]
        if raw is None:
            self.cmd(*unlock)
            raise SessionError("no such session", 404)
        s = Session.from_dict(json.loads(raw))
        try:
            yield s
        except BaseException:
            self.cmd(*unlock)
            raise
        self.pipeline(["SET", f"session:{sid}", json.dumps(s.to_dict()), "EX", SESSION_TTL_S], unlock)

    def rtt_ms(self) -> float:
        t = time.perf_counter()
        self.cmd("PING")
        return round((time.perf_counter() - t) * 1000, 1)

    def claim(self, key: str) -> bool:
        return bool(self.cmd("SET", f"claim:{key}", "1", "NX", "EX", 86400))


def make_store() -> Store:
    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    return RedisStore(url, token) if url and token else Store()
