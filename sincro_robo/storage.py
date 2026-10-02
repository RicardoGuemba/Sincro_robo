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
                CREATE TABLE IF NOT EXISTS calibration_campaigns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_captures (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES calibration_campaigns(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    corners_json TEXT,
                    plan_z REAL,
                    cell_json TEXT,
                    reprojection_px REAL,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_campaign_captures
                ON campaign_captures(campaign_id, stage);
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

    def merge_session_config(self, session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        config = dict(session["config"])
        config.update(patch)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET config_json = ? WHERE id = ?",
                (json.dumps(config, ensure_ascii=False), session_id),
            )
        updated = self.get_session(session_id)
        assert updated is not None
        return updated

    def mark_profiles_stale(self, except_hash: str) -> None:
        sessions = self.list_sessions()
        for session in sessions:
            profile = session["config"].get("optical_profile")
            if not profile or profile.get("profile_sha256") == except_hash or profile.get("stale"):
                continue
            profile = dict(profile)
            profile["stale"] = True
            self.merge_session_config(session["id"], {"optical_profile": profile})

    def create_campaign(self, name: str, state: dict[str, Any]) -> dict[str, Any]:
        campaign_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO calibration_campaigns
                (id, name, created_at, stage, status, state_json)
                VALUES (?, ?, ?, ?, 'active', ?)
                """,
                (campaign_id, name, utc_now(), state["stage"], json.dumps(state, ensure_ascii=False)),
            )
        self.audit(campaign_id, "campaign_created", {"name": name})
        campaign = self.get_campaign(campaign_id)
        assert campaign is not None
        return campaign

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calibration_campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        result["captures"] = self.list_campaign_captures(campaign_id)
        return result

    def list_campaigns(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM calibration_campaigns ORDER BY created_at DESC"
            ).fetchall()
        return [campaign for row in rows if (campaign := self.get_campaign(row["id"]))]

    def save_campaign_state(self, campaign_id: str, stage: str, state: dict[str, Any], status: str = "active") -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE calibration_campaigns
                SET stage = ?, status = ?, state_json = ?
                WHERE id = ?
                """,
                (stage, status, json.dumps(state, ensure_ascii=False), campaign_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(campaign_id)

    def add_campaign_capture(self, campaign_id: str, capture: dict[str, Any]) -> dict[str, Any]:
        capture_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO campaign_captures
                (id, campaign_id, stage, decision, reason, corners_json, plan_z, cell_json,
                 reprojection_px, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    campaign_id,
                    capture["stage"],
                    capture["decision"],
                    capture.get("reason") or "",
                    json.dumps(capture.get("corners") or [], ensure_ascii=False),
                    capture.get("plan_z"),
                    json.dumps(capture.get("cell"), ensure_ascii=False) if capture.get("cell") else None,
                    capture.get("reprojection_px"),
                    json.dumps(capture["metrics"], ensure_ascii=False) if capture.get("metrics") else None,
                    utc_now(),
                ),
            )
        self.audit(campaign_id, "campaign_capture", {"capture_id": capture_id, "decision": capture["decision"]})
        return self._capture_by_id(capture_id)

    def update_campaign_capture(self, capture_id: str, **fields: Any) -> None:
        allowed = {"decision", "reason", "reprojection_px", "metrics", "corners"}
        column = {
            "decision": "decision",
            "reason": "reason",
            "reprojection_px": "reprojection_px",
            "metrics": "metrics_json",
            "corners": "corners_json",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            stored = value
            if key in {"metrics", "corners"}:
                stored = json.dumps(value, ensure_ascii=False) if value is not None else None
            assignments.append(f"{column[key]} = ?")
            values.append(stored)
        if not assignments:
            return
        values.append(capture_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE campaign_captures SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    def delete_campaign_captures(self, campaign_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM campaign_captures WHERE campaign_id = ?", (campaign_id,))

    def list_campaign_captures(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM campaign_captures
                WHERE campaign_id = ? ORDER BY created_at
                """,
                (campaign_id,),
            ).fetchall()
        return [self._decode_capture(row) for row in rows]

    def _capture_by_id(self, capture_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_captures WHERE id = ?", (capture_id,)
            ).fetchone()
        if row is None:
            raise KeyError(capture_id)
        return self._decode_capture(row)

    @staticmethod
    def _decode_capture(row: sqlite3.Row) -> dict[str, Any]:
        corners = json.loads(row["corners_json"]) if row["corners_json"] else []
        cell = json.loads(row["cell_json"]) if row["cell_json"] else None
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        return {
            "id": row["id"],
            "campaign_id": row["campaign_id"],
            "stage": row["stage"],
            "decision": row["decision"],
            "reason": row["reason"] or "",
            "corners": corners,
            "corner_count": len(corners),
            "plan_z": row["plan_z"],
            "cell": cell,
            "reprojection_px": row["reprojection_px"],
            "metrics": metrics,
            "created_at": row["created_at"],
        }

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
