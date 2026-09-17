from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..calibration import evaluate_plan, fit_plan_calibration
from ..domain import CaptureCandidate, RobotPoseSnapshot, VisionObservation, utc_now
from ..geometry import configured_pixel_scales, signed_axis_delta
from ..storage import Storage


POINT_REGIONS = (
    ("Centro", "adjustment", (0.50, 0.50)),
    ("Superior esquerda", "adjustment", (0.22, 0.24)),
    ("Superior direita", "adjustment", (0.78, 0.24)),
    ("Inferior esquerda", "adjustment", (0.22, 0.76)),
    ("Inferior direita", "adjustment", (0.78, 0.76)),
    ("Validação esquerda", "validation", (0.14, 0.50)),
    ("Validação direita", "validation", (0.86, 0.50)),
    ("Expansão superior", "adjustment", (0.50, 0.14)),
    ("Expansão inferior", "adjustment", (0.50, 0.86)),
)


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except Exception:
        return "unversioned"


class CaptureController:
    def __init__(
        self,
        storage: Storage,
        config: dict[str, Any],
        vision_supplier: Callable[[], VisionObservation | None],
        robot_supplier: Callable[[], RobotPoseSnapshot | None],
        project_root: Path,
    ) -> None:
        self.storage = storage
        self.config = config
        self.vision_supplier = vision_supplier
        self.robot_supplier = robot_supplier
        self.project_root = project_root
        self.active_session_id: str | None = None
        self.active_plan_z: float | None = None
        self._candidate: CaptureCandidate | None = None
        self._lock = threading.RLock()
        self.model_hash = sha256_file(config["model"]["checkpoint"])
        self.revision = git_revision(project_root)

    @property
    def candidate(self) -> CaptureCandidate | None:
        with self._lock:
            return self._candidate

    def create_session(self, name: str, planes: list[float]) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Informe o nome da sessão")
        if not planes:
            raise ValueError("A sessão precisa de pelo menos um plano Z")
        normalized = [float(value) for value in planes]
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError("Os planos Z devem ser números finitos")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Os planos Z não podem ser duplicados")
        calibration = self.config["calibration"]
        pixel_reference = self.config["pixel_reference"]
        scale_x, scale_y = configured_pixel_scales(pixel_reference)
        snapshot = {
            "xy_tolerance_mm": calibration["xy_tolerance_mm"],
            "angular_tolerance_deg": calibration["angular_tolerance_deg"],
            "pick_offset_local_mm": calibration["pick_offset_local_mm"],
            "default_points_per_plane": calibration["default_points_per_plane"],
            "max_points_per_plane": calibration["max_points_per_plane"],
            "model_threshold": self.config["model"]["threshold"],
            "vision_quality": self.config["vision_quality"],
            "source_config": self.config["_config_path"],
            "pixel_reference": {
                "source_width": int(pixel_reference["source_width"]),
                "source_height": int(pixel_reference["source_height"]),
                "destination_width": int(pixel_reference["destination_width"]),
                "destination_height": int(pixel_reference["destination_height"]),
                "scale_x": scale_x,
                "scale_y": scale_y,
            },
        }
        return self.storage.create_session(
            clean_name,
            normalized,
            __version__,
            self.model_hash,
            self.revision,
            snapshot,
        )

    def activate_session(self, session_id: str) -> dict[str, Any]:
        session = self.storage.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        self.storage.start_session(session_id)
        with self._lock:
            self.active_session_id = session_id
            self.active_plan_z = None
            self._candidate = None
        return self.session_detail(session_id)

    def activate_plan(self, session_id: str, plan_z: float) -> dict[str, Any]:
        session = self.storage.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        value = float(plan_z)
        if value not in [float(item) for item in session["config"]["planes_mm"]]:
            raise ValueError("Plano Z não pertence à sessão")
        if self.active_session_id != session_id:
            self.activate_session(session_id)
        with self._lock:
            self.active_plan_z = value
            self._candidate = None
        self.storage.audit(session_id, "plan_activated", {"plan_z": value})
        return self.session_detail(session_id)

    def _next_point(self, session_id: str, plan_z: float) -> tuple[int, str, str, tuple[float, float]]:
        count = len(self.storage.list_pairs(session_id, plan_z))
        index = count + 1
        if index > len(POINT_REGIONS):
            raise ValueError("O plano já atingiu o máximo de 9 pontos")
        region, role, target = POINT_REGIONS[index - 1]
        return index, region, role, target

    def _image_normalized(self, observation: VisionObservation) -> tuple[float, float]:
        overlay_x, overlay_y = observation.overlay_xy()
        return (
            overlay_x / max(1.0, float(observation.frame_width)),
            overlay_y / max(1.0, float(observation.frame_height)),
        )

    def _check_duplicate(
        self, session_id: str, plan_z: float, observation: VisionObservation
    ) -> None:
        calibration = self.config["calibration"]
        for pair in self.storage.list_pairs(session_id, plan_z):
            vision = pair["vision"]
            distance = math.hypot(observation.x - vision["x"], observation.y - vision["y"])
            angle = abs(signed_axis_delta(observation.angle_deg, vision["angle_deg"]))
            if (
                distance < calibration["duplicate_distance_px"]
                and angle < calibration["duplicate_angle_deg"]
            ):
                raise ValueError("Ponto duplicado: mova o molde para a região sugerida")

    def _is_duplicate(
        self, session_id: str, plan_z: float, observation: VisionObservation
    ) -> bool:
        try:
            self._check_duplicate(session_id, plan_z, observation)
            return False
        except ValueError:
            return True

    def capture_readiness(self) -> dict[str, Any]:
        with self._lock:
            session_id = self.active_session_id
            plan_z = self.active_plan_z
            candidate_pending = self._candidate is not None
        vision = self.vision_supplier()
        robot = self.robot_supplier()
        result: dict[str, Any] = {
            "session": session_id is not None,
            "plan": plan_z is not None,
            "single_instance": bool(vision and vision.gates.get("single_instance")),
            "confidence": bool(vision and vision.gates.get("confidence")),
            "mask_not_cut": bool(vision and vision.gates.get("mask_not_cut")),
            "axis_quality": bool(vision and vision.gates.get("axis_quality")),
            "mask_area": bool(vision and vision.gates.get("mask_area")),
            "stable": bool(vision and vision.stable),
            "pose": bool(robot and robot.fresh),
            "region": False,
            "plan_z": False,
            "not_duplicate": False,
            "candidate_clear": not candidate_pending,
        }
        if session_id is None or plan_z is None or vision is None:
            result["ready"] = False
            return result
        try:
            _index, _region, _role, target = self._next_point(session_id, plan_z)
        except ValueError:
            result["ready"] = False
            return result
        normalized_x, normalized_y = self._image_normalized(vision)
        radius = float(self.config["calibration"]["target_region_radius_norm"])
        result["region"] = math.hypot(normalized_x - target[0], normalized_y - target[1]) <= radius
        result["not_duplicate"] = not self._is_duplicate(session_id, plan_z, vision)
        if robot is not None:
            tolerance = float(self.config["calibration"]["plan_z_tolerance_mm"])
            result["plan_z"] = abs(robot.z - plan_z) <= tolerance
        result["ready"] = result["session"] and result["plan"] and result["candidate_clear"] and robot is not None
        return result

    def capture(self) -> CaptureCandidate:
        with self._lock:
            if self._candidate is not None:
                raise ValueError("Já existe um candidato aguardando confirmação")
            session_id = self.active_session_id
            plan_z = self.active_plan_z
        if session_id is None or plan_z is None:
            raise ValueError("Ative uma sessão e um plano Z antes da captura")
        vision = self.vision_supplier()
        robot = self.robot_supplier()
        if vision is None:
            raise ValueError("Visão indisponível")
        if robot is None:
            raise ValueError("Pose do robô indisponível")
        point_index, region, role, _target = self._next_point(session_id, plan_z)
        candidate = CaptureCandidate(
            id=str(uuid.uuid4()),
            session_id=session_id,
            plan_z=plan_z,
            point_index=point_index,
            role=role,
            region=region,
            vision=vision,
            robot=robot,
            created_at=utc_now(),
        )
        with self._lock:
            self._candidate = candidate
        self.storage.audit(
            session_id,
            "capture_frozen",
            {"candidate_id": candidate.id, "plan_z": plan_z, "point": point_index},
        )
        return candidate

    def decide(self, confirm: bool) -> dict[str, Any]:
        with self._lock:
            candidate = self._candidate
            if candidate is None:
                raise ValueError("Não há candidato aguardando confirmação")
            self._candidate = None
        if not confirm:
            self.storage.audit(
                candidate.session_id,
                "capture_cancelled",
                {"candidate_id": candidate.id, "plan_z": candidate.plan_z},
            )
            return {"saved": False, "candidate_id": candidate.id}
        saved = self.storage.save_candidate(candidate)
        evaluation = self.evaluate(candidate.session_id, candidate.plan_z, persist_results=True)
        self._maybe_expand(candidate.session_id, candidate.plan_z, evaluation)
        return {"saved": True, "pair": saved, "evaluation": evaluation}

    def evaluate(
        self,
        session_id: str,
        plan_z: float,
        persist_results: bool = False,
    ) -> dict[str, Any]:
        pairs = self.storage.list_pairs(session_id, plan_z)
        adjustment_count = sum(pair["role"] == "adjustment" for pair in pairs)
        if adjustment_count < 3:
            return {
                "ready": False,
                "reason": "mínimo de 3 pontos de ajuste ainda não atingido",
                "adjustment_count": adjustment_count,
            }
        calibration = self.config["calibration"]
        try:
            model = fit_plan_calibration(
                plan_z, pairs, calibration["pick_offset_local_mm"]
            )
        except ValueError as error:
            return {"ready": False, "reason": str(error), "adjustment_count": adjustment_count}
        metrics = evaluate_plan(
            model,
            pairs,
            calibration["pick_offset_local_mm"],
            calibration["xy_tolerance_mm"],
            calibration["angular_tolerance_deg"],
        )
        if persist_results:
            for pair_result in metrics["pairs"]:
                self.storage.update_pair_result(
                    pair_result["pair_id"],
                    pair_result["status"],
                    pair_result["residual_xy_mm"],
                    pair_result["error_angle_deg"],
                )
        return {"ready": True, "model": model.to_dict(), "metrics": metrics}

    def _maybe_expand(self, session_id: str, plan_z: float, evaluation: dict[str, Any]) -> None:
        pairs = self.storage.list_pairs(session_id, plan_z)
        count = len(pairs)
        if count < 7 or not evaluation.get("ready"):
            return
        session = self.storage.get_session(session_id)
        assert session is not None
        targets = session["config"].get("plan_targets", {})
        current_target = int(targets.get(str(float(plan_z)), 7))
        passed = bool(evaluation.get("metrics", {}).get("passed"))
        max_points = int(self.config["calibration"]["max_points_per_plane"])
        if not passed and count >= current_target and current_target < max_points:
            self.storage.update_plan_target(session_id, plan_z, current_target + 1)

    def session_detail(self, session_id: str) -> dict[str, Any]:
        session = self.storage.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        pairs = self.storage.list_pairs(session_id)
        plans: list[dict[str, Any]] = []
        for plan_z in session["config"]["planes_mm"]:
            plan_pairs = [pair for pair in pairs if float(pair["plan_z"]) == float(plan_z)]
            target = int(session["config"].get("plan_targets", {}).get(str(float(plan_z)), 7))
            next_point = len(plan_pairs) + 1
            suggestion = None
            if next_point <= min(target, len(POINT_REGIONS)):
                region, role, normalized = POINT_REGIONS[next_point - 1]
                suggestion = {"index": next_point, "region": region, "role": role, "normalized": normalized}
            plans.append(
                {
                    "z": float(plan_z),
                    "count": len(plan_pairs),
                    "target": target,
                    "complete": len(plan_pairs) >= target,
                    "suggestion": suggestion,
                    "evaluation": self.evaluate(session_id, float(plan_z)),
                }
            )
        session["pairs"] = pairs
        session["plans"] = plans
        session["active"] = session_id == self.active_session_id
        session["active_plan_z"] = self.active_plan_z if session["active"] else None
        return session

    def export_session(self, session_id: str, exports_dir: str) -> dict[str, str]:
        detail = self.session_detail(session_id)
        audit = self.storage.list_audit(session_id)
        export_root = Path(exports_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        stem = f"sincro_robo_{session_id[:8]}"
        json_path = export_root / f"{stem}.json"
        csv_path = export_root / f"{stem}_pontos.csv"
        payload = {
            "schema_version": "1",
            "exported_at": utc_now(),
            "session": detail,
            "audit": audit,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = [
                "id", "plan_z", "point_index", "role", "region", "saved_at", "status",
                "vision_x", "vision_y", "vision_angle_deg", "confidence", "axis_quality",
                "robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry", "robot_rz",
                "residual_xy_mm", "error_angle_deg",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for pair in detail["pairs"]:
                writer.writerow(
                    {
                        "id": pair["id"], "plan_z": pair["plan_z"],
                        "point_index": pair["point_index"], "role": pair["role"],
                        "region": pair["region"], "saved_at": pair["saved_at"],
                        "status": pair["status"], "vision_x": pair["vision"]["x"],
                        "vision_y": pair["vision"]["y"],
                        "vision_angle_deg": pair["vision"]["angle_deg"],
                        "confidence": pair["vision"]["confidence"],
                        "axis_quality": pair["vision"]["axis_quality"],
                        "robot_x": pair["robot"]["x"], "robot_y": pair["robot"]["y"],
                        "robot_z": pair["robot"]["z"], "robot_rx": pair["robot"]["rx"],
                        "robot_ry": pair["robot"]["ry"], "robot_rz": pair["robot"]["rz"],
                        "residual_xy_mm": pair["residual_xy"],
                        "error_angle_deg": pair["residual_angle"],
                    }
                )
        self.storage.audit(
            session_id,
            "session_exported",
            {"json": json_path.name, "csv": csv_path.name},
        )
        return {"json": json_path.name, "csv": csv_path.name}
