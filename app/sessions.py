"""In-memory sessions. All timing comes from the server's monotonic clock.

The clock starts when the puzzle is *served* (GET) and stops when the answer arrives (POST). Network
round-trip is included, which only ever makes a solver look slower, never faster, so it can't
create a false "too fast" flag. The client's own timer is recorded but never trusted.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from app import puzzles
from app.detector import Round

PUZZLE_TTL_S = 60.0
DEFAULT_ROUNDS = 6


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
    ask_declaration: bool = True
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
                if d is None or time.monotonic() - d.issued_at > PUZZLE_TTL_S:
                    self.declaration = d = RoundRecord(puzzle=puzzles.declaration(),
                                                       issued_at=time.monotonic())
                return {**d.puzzle.public(), "round": 0, "rounds": self.total_rounds}
            cur = self.current()
            if cur is None or time.monotonic() - cur.issued_at > PUZZLE_TTL_S:
                # A fresh puzzle each time. Re-fetching an open puzzle returns it with the clock
                # still running, so refreshing never buys time.
                if cur is not None:
                    self.records.pop()
                cur = RoundRecord(puzzle=puzzles.generate(), issued_at=time.monotonic())
                self.records.append(cur)
            return {**cur.puzzle.public(), "round": len(self.records), "rounds": self.total_rounds}

    def answer(self, puzzle_id: str, choice: str, telemetry: dict | None) -> dict:
        now = time.monotonic()
        with self.lock:
            cur = self.current()
            if cur is None or cur.puzzle.id != puzzle_id:
                raise SessionError("unknown or already-answered puzzle", 409)
            if now - cur.issued_at > PUZZLE_TTL_S:
                raise SessionError("puzzle expired", 410)
            if choice not in cur.puzzle.words:
                raise SessionError("choice is not one of the options", 422)
            cur.answered_at, cur.choice = now, choice
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


class Store:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, actor: str, rounds: int = DEFAULT_ROUNDS, declaration: bool = True) -> Session:
        s = Session(id=secrets.token_urlsafe(9), actor=actor, total_rounds=rounds,
                    ask_declaration=declaration)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if s is None:
            raise SessionError("no such session", 404)
        return s
