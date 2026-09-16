from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sincro_robo.api import create_app
from sincro_robo.config import load_config


def test_health_and_session_api(tmp_path: Path) -> None:
    config = load_config("config/simulator.json")
    config["app"]["database"] = str(tmp_path / "api.sqlite3")
    config["app"]["exports_dir"] = str(tmp_path / "exports")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    with TestClient(create_app(config)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["version"] == "0.4.0"

        created = client.post(
            "/api/sessions", json={"name": "API local", "planes": [0, 200, 400]}
        )
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert client.post(f"/api/sessions/{session_id}/activate").status_code == 200
        detail = client.post(f"/api/sessions/{session_id}/plans/0/activate")
        assert detail.status_code == 200
        assert detail.json()["active_plan_z"] == 0.0


def test_session_requires_a_plan(tmp_path: Path) -> None:
    config = load_config("config/simulator.json")
    config["app"]["database"] = str(tmp_path / "api.sqlite3")
    config["app"]["exports_dir"] = str(tmp_path / "exports")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    with TestClient(create_app(config)) as client:
        response = client.post("/api/sessions", json={"name": "Inválida", "planes": []})
        assert response.status_code == 422

