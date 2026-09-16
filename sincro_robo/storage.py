from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .domain import CaptureCandidate, utc_now


class Storage:
    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    status TEXT NOT NULL,
                    app_version TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    git_revision TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pairs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    plan_z REAL NOT NULL,
                    point_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    region TEXT NOT NULL,
                    vision_json TEXT NOT NULL,
                    robot_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    residual_xy REAL,
                    residual_angle REAL,
                    UNIQUE(session_id, plan_z, point_index)
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pairs_session_plan
                ON pairs(session_id, plan_z, point_index);
                """
            )

    @staticmethod
    def _decode_session(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    @staticmethod
    def _decode_pair(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["vision"] = json.loads(result.pop("vision_json"))
        result["robot"] = json.loads(result.pop("robot_json"))
        return result

    def create_session(
        self,
        name: str,
        planes: list[float],
        app_version: str,
        model_hash: str,
        git_revision: str,
        config_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        session_config = dict(config_snapshot)
        session_config["planes_mm"] = planes
        default_target = int(session_config.get("default_points_per_plane", 7))
        session_config["plan_targets"] = {str(value): default_target for value in planes}
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions
                (id, name, created_at, status, app_version, model_hash, git_revision, config_json)
                VALUES (?, ?, ?, 'draft', ?, ?, ?, ?)
                """,
                (
                    session_id,
                    name,
                    utc_now(),
                    app_version,
                    model_hash,
                    git_revision,
                    json.dumps(session_config, ensure_ascii=False),
                ),
            )
        self.audit(session_id, "session_created", {"name": name, "planes_mm": planes})
        session = self.get_session(session_id)
        assert session is not None
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [self._decode_session(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._decode_session(row) if row else None

    def start_session(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = 'active', started_at = COALESCE(started_at, ?)
                WHERE id = ?
                """,
                (utc_now(), session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(session_id)
        self.audit(session_id, "session_activated", {})

    def update_plan_target(self, session_id: str, plan_z: float, target: int) -> None:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        config = session["config"]
        config.setdefault("plan_targets", {})[str(float(plan_z))] = int(target)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET config_json = ? WHERE id = ?",
                (json.dumps(config, ensure_ascii=False), session_id),
            )
        self.audit(session_id, "plan_target_changed", {"plan_z": plan_z, "target": target})

    def list_pairs(self, session_id: str, plan_z: float | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM pairs WHERE session_id = ?"
        params: list[Any] = [session_id]
        if plan_z is not None:
            query += " AND plan_z = ?"
            params.append(float(plan_z))
        query += " ORDER BY plan_z, point_index"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._decode_pair(row) for row in rows]

    def save_candidate(self, candidate: CaptureCandidate) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pairs
                (id, session_id, plan_z, point_index, role, region,
                 vision_json, robot_json, saved_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')
                """,
                (
                    candidate.id,
                    candidate.session_id,
                    candidate.plan_z,
                    candidate.point_index,
                    candidate.role,
                    candidate.region,
                    json.dumps(candidate.vision.to_dict(), ensure_ascii=False),
                    json.dumps(candidate.robot.to_dict(), ensure_ascii=False),
                    utc_now(),
                ),
            )
        self.audit(
            candidate.session_id,
            "capture_confirmed",
            {"pair_id": candidate.id, "plan_z": candidate.plan_z, "point": candidate.point_index},
        )
        return self.list_pairs(candidate.session_id, candidate.plan_z)[-1]

    def update_pair_result(
        self,
        pair_id: str,
        status: str,
        residual_xy: float,
        residual_angle: float,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE pairs SET status = ?, residual_xy = ?, residual_angle = ?
                WHERE id = ?
                """,
                (status, residual_xy, residual_angle, pair_id),
            )

    def audit(self, session_id: str | None, event: str, detail: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO audit(session_id, timestamp, event, detail_json) VALUES (?, ?, ?, ?)",
                (session_id, utc_now(), event, json.dumps(detail, ensure_ascii=False)),
            )

    def list_audit(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp, event, detail_json FROM audit WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"timestamp": row["timestamp"], "event": row["event"], "detail": json.loads(row["detail_json"])}
            for row in rows
        ]
