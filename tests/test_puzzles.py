import random
import time
from collections import Counter

import pytest

from app import puzzles, sessions
from app.sessions import SessionError, Store


def test_no_word_in_two_categories():
    counts = Counter(w for ws in puzzles.CATEGORIES.values() for w in ws)
    assert [w for w, c in counts.items() if c > 1] == []


def test_puzzle_shape_and_answer():
    rng = random.Random(0)
    for _ in range(500):
        p = puzzles.generate(rng)
        assert len(p.words) == len(set(p.words)) == puzzles.N_OPTIONS
        assert p.answer in p.words and p.answer in puzzles.CATEGORIES[p.intruder_category]
        assert sum(w in puzzles.CATEGORIES[p.category] for w in p.words) == 4
        assert not puzzles._related(p.category, p.intruder_category)


def test_answer_never_in_public_payload():
    p = puzzles.generate()
    pub = p.public()
    assert set(pub) == {"puzzle_id", "kind", "prompt", "options", "words"}
    # Nothing marks which option is the intruder, and the category names don't leak either.
    flat = str(pub)
    assert p.category not in flat and p.intruder_category not in flat
    assert sorted(pub["options"]) == sorted(p.words)


def test_puzzles_are_fresh():
    assert len({tuple(puzzles.generate().words) for _ in range(200)}) > 195


def test_replay_and_wrong_puzzle_rejected():
    s = Store().create("web", rounds=3, declaration=False)
    p = s.next_puzzle()
    s.answer(p["puzzle_id"], p["words"][0], {})
    with pytest.raises(SessionError):
        s.answer(p["puzzle_id"], p["words"][0], {})  # replay
    q = s.next_puzzle()
    with pytest.raises(SessionError):
        s.answer(q["puzzle_id"], "not-an-option", {})


def test_refetch_keeps_the_clock_running():
    s = Store().create("web", rounds=3, declaration=False)
    a = s.next_puzzle()
    time.sleep(0.05)
    b = s.next_puzzle()
    assert a["puzzle_id"] == b["puzzle_id"]
    s.answer(b["puzzle_id"], b["words"][0], {})
    assert s.answered[0].latency_s >= 0.05


def test_expired_puzzle_rejected(monkeypatch):
    s = Store().create("web", rounds=3, declaration=False)
    p = s.next_puzzle()
    monkeypatch.setattr(sessions, "PUZZLE_TTL_S", 0.0)
    with pytest.raises(SessionError) as e:
        s.answer(p["puzzle_id"], p["words"][0], {})
    assert e.value.status == 410


def test_session_completes():
    s = Store().create("web", rounds=3, declaration=False)
    for _ in range(3):
        p = s.next_puzzle()
        s.answer(p["puzzle_id"], p["words"][0], {})
    assert s.done
    with pytest.raises(SessionError):
        s.next_puzzle()


def test_declaration_comes_first_and_is_not_a_round():
    s = Store().create("web", rounds=3)
    d = s.next_puzzle()
    assert d["kind"] == "declaration" and d["round"] == 0
    assert sorted(d["options"]) == sorted([puzzles.DECLARE_AGENT, puzzles.DECLARE_HUMAN])
    s.answer(d["puzzle_id"], puzzles.DECLARE_AGENT, {})
    assert s.declared()["choice"] == "agent" and s.answered == []
    p = s.next_puzzle()
    assert p["kind"] == "odd_one_out" and p["round"] == 1
