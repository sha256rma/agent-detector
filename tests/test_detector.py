import random

import pytest

from app.detector import (CHANCE, DEFAULT_HUMAN_FLOOR_S, Round, binom_tail, classify,
                          human_floor_from)


def session(latencies, correct, events=0):
    return [Round(latency_s=t, correct=c, pointer_moves=events) for t, c in zip(latencies, correct)]


@pytest.mark.parametrize("lat,correct,expected", [
    ([0.004] * 6, [1, 0, 0, 0, 1, 0], "software_bot"),          # instant, at chance
    ([0.18, 0.21, 0.19, 0.2, 0.22, 0.17], [1, 1, 1, 1, 0, 1], "llm_agent"),
    ([2.8, 3.4, 2.1, 4.0, 2.6, 3.1], [1, 1, 1, 1, 1, 0], "human"),
    ([3.1, 2.9, 3.5, 2.7, 3.3, 4.1], [1, 0, 0, 0, 0, 1], "inconclusive"),
    ([0.001] * 6, [1, 1, 0, 1, 0, 0], "software_bot"),           # lucky 3/6 guesser stays a bot
])
def test_truth_table(lat, correct, expected):
    assert classify(session(lat, correct))["label"] == expected


def test_stalling_agent_passes_as_human():
    """The one-sided limit, pinned as a test: delay alone moves an agent into the human cell."""
    v = classify(session([2.5, 3.0, 3.8, 2.2, 3.4, 2.9], [1] * 6, events=0))
    assert v["label"] == "human"
    assert any("usually report none" in r for r in v["reasons"])
    assert any("slows down and lies" in r for r in v["reasons"])


def test_scores_are_a_distribution():
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(3, 12)
        v = classify(session([rng.uniform(0.001, 10) for _ in range(n)], [rng.random() < .5 for _ in range(n)]))
        assert sum(v["scores"].values()) == pytest.approx(1, abs=1e-3)
        assert max(v["scores"], key=v["scores"].get) == v["label"]


def test_binomial_tail():
    assert binom_tail(0, 6, CHANCE) == pytest.approx(1)
    assert binom_tail(6, 6, CHANCE) == pytest.approx(0.2**6)
    assert binom_tail(4, 6, CHANCE) < 0.05 < binom_tail(3, 6, CHANCE)


def test_floor_is_the_budget_quantile():
    lats = [i / 10 for i in range(1, 101)]  # 0.1 .. 10.0
    assert human_floor_from(lats, budget=0.05) == pytest.approx(0.5)
    assert human_floor_from(lats[:5]) == DEFAULT_HUMAN_FLOOR_S  # too few samples: default


def test_empty_session():
    assert classify([])["label"] is None


SLOW_ACCURATE = ([2.5, 3.0, 3.8, 2.2, 3.4, 2.9], [1] * 6)


def test_declared_agent_catches_the_staller():
    v = classify(session(*SLOW_ACCURATE), declared="agent")
    assert v["label"] == "llm_agent" and v["declaration_used"]
    assert v["scores"]["human"] == 0
    assert any("slows down on purpose" in r for r in v["reasons"])


def test_declared_human_changes_nothing():
    assert classify(session(*SLOW_ACCURATE), declared="human")["label"] == "human"
    fast = classify(session([0.2] * 6, [1] * 6), declared="human")
    assert fast["label"] == "llm_agent" and any("not true" in r for r in fast["reasons"])


def test_declaration_from_a_guesser_is_ignored():
    v = classify(session([0.002] * 6, [1, 0, 0, 0, 0, 0]), declared="agent")
    assert v["label"] == "software_bot" and not v["declaration_used"]
