from __future__ import annotations

from fastapi.testclient import TestClient

from tailsafe.api.app import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "not a substitute" in body["disclaimer"]


def test_params_endpoint_lists_sources() -> None:
    res = client.get("/api/params")
    assert res.status_code == 200
    rows = res.json()
    assert rows and all(r["source"] for r in rows)
