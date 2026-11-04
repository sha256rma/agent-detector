"""The calibration endpoint sets the speed threshold, so it must reject runs that could poison it."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "HUMAN_RUNS", tmp_path / "human_runs.jsonl")
    return TestClient(main.app)  # no `with`: skips loading the model at startup


def play(client, latency_ok=True, events=5):
    sid = client.post("/api/session", json={"rounds": 3}).json()["session_id"]
    s = main.store.get(sid)
    while not s.done:
        p = client.get(f"/api/session/{sid}/puzzle").json()
        client.post(f"/api/session/{sid}/answer", json={
            "puzzle_id": p["puzzle_id"], "choice": p["options"][0],
            "telemetry": {"pointer_moves": events, "keys": 0}})
    if latency_ok:  # pretend each answer took a human-like 2 s
        for r in s.answered:
            r.answered_at = r.issued_at + 2.0
    return sid


def test_genuine_run_saves_once(client):
    sid = play(client)
    assert client.post(f"/api/session/{sid}/label-human").status_code == 200
    assert client.post(f"/api/session/{sid}/label-human").status_code == 409


def test_cross_site_request_rejected(client):
    sid = play(client)
    r = client.post(f"/api/session/{sid}/label-human", headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_instant_answers_rejected(client):
    assert client.post(f"/api/session/{play(client, latency_ok=False)}/label-human").status_code == 422


def test_no_interaction_rejected(client):
    assert client.post(f"/api/session/{play(client, events=0)}/label-human").status_code == 422
