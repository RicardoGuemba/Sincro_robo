"""Assistente da campanha. A HTTP só chama estes métodos; a câmera só observa o frame."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calibration import fit_plan_calibration
from ..campaign_flow import BOARD_STAGES, STAGE_GUIDE, STAGES, coverage_view, gate_for, next_stage
from ..domain import utc_now
from ..optics import (
    PATTERN,
    VALIDATION_PLANES_MM,
    board_frame,
    calibrate_intrinsic,
    canonical_hash,
    coverage_cell,
    assess_focus,
    detect_chessboard,
    diagnose_residuals,
    direction_errors,
    draw_detection,
    preview_before_after,
    sharpness_score,
    robot_from_camera,
    solve_extrinsic,
    undistort_pairs,
    validation_metrics,
)
from ..storage import Storage
from .capture import CaptureController


class CampaignController:
    def __init__(
        self,
        storage: Storage,
        config: dict[str, Any],
        capture: CaptureController,
        project_root: Path,
    ) -> None:
        self.storage = storage
        self.config = config
        self.capture = capture
        self.project_root = project_root
        self._lock = threading.RLock()
        self.active_id: str | None = None
        self.live_detection: dict[str, Any] | None = None
        self._scene = "mold"
        self._pose_override: tuple[float, float, float] | None = None
        self._reset_focus_tracker()

    def camera_scene(self) -> str:
        with self._lock:
            return self._scene

    def pose_override(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._pose_override

    def create(self, name: str, square_size_mm: float) -> dict[str, Any]:
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 120:
            raise ValueError("Dê um nome de 1 a 120 caracteres para a campanha")
        if float(square_size_mm) <= 0:
            raise ValueError("O lado do quadrado precisa ser maior que zero")
        state = {
            "stage": "fix_hardware",
            "square_size_mm": float(square_size_mm),
            "pattern": list(PATTERN),
            "checks": {
                "camera_fixed": False,
                "lens_focus_locked": False,
                "production_resolution": False,
                "single_stapi_client": False,
            },
            "fingerprint": None,
            "image_size_px": None,
            "intrinsic": None,
            "reprojection_reviewed": False,
            "undistort_confirmed": False,
            "frame_points": {},
            "board_frame": None,
            "extrinsic": None,
            "direction": None,
            "direction_check": None,
            "validation_plan_z": 0.0,
            "affine_session_id": None,
            "profile_sha256": None,
            "focus_lock": None,
            "focus_drift": False,
            "created_at": utc_now(),
        }
        campaign = self.storage.create_campaign(cleaned, state)
        with self._lock:
            self.active_id = campaign["id"]
        self._reset_focus_tracker()
        self._publish_simulation(campaign)
        return self.view()

    def set_checks(self, checks: dict[str, bool]) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "fix_hardware")
        self._require_focus_held(campaign)
        state = campaign["state"]
        current = dict(state["checks"])
        for key in current:
            if key in checks:
                current[key] = bool(checks[key])
        if current["lens_focus_locked"] and not self._focus_is_stable():
            current["lens_focus_locked"] = False
            state["checks"] = current
            state["focus_lock"] = None
            self._save(campaign)
            raise ValueError("Estabilize a nitidez no máximo, com o tabuleiro visível, antes de travar o foco.")
        state["checks"] = current
        if current["lens_focus_locked"]:
            state["focus_lock"] = {"peak": float(self._focus_peak), "locked_at": utc_now()}
        else:
            state["focus_lock"] = None
            self._focus_below = 0
        self._save(campaign)
        return self.view()

    def restart(self) -> dict[str, Any]:
        campaign = self._require()
        if not campaign["state"].get("focus_drift"):
            raise ValueError("O foco travado ainda está estável.")
        square = float(campaign["state"]["square_size_mm"])
        pattern = list(campaign["state"]["pattern"])
        self.storage.delete_campaign_captures(campaign["id"])
        if campaign["state"].get("profile_sha256"):
            self.storage.mark_profiles_stale("focus-restart")
        campaign["stage"] = "fix_hardware"
        campaign["state"] = {
            "stage": "fix_hardware",
            "square_size_mm": square,
            "pattern": pattern,
            "checks": {
                "camera_fixed": False,
                "lens_focus_locked": False,
                "production_resolution": False,
                "single_stapi_client": False,
            },
            "fingerprint": None,
            "image_size_px": None,
            "intrinsic": None,
            "reprojection_reviewed": False,
            "undistort_confirmed": False,
            "frame_points": {},
            "board_frame": None,
            "extrinsic": None,
            "direction": None,
            "direction_check": None,
            "validation_plan_z": 0.0,
            "affine_session_id": None,
            "profile": None,
            "profile_sha256": None,
            "focus_lock": None,
            "focus_drift": False,
        }
        campaign["captures"] = []
        self._reset_focus_tracker()
        self._save(campaign)
        self.storage.audit(campaign["id"], "campaign_restarted", {"reason": "focus_drift"})
        return self.view()

    def advance(self) -> dict[str, Any]:
        campaign = self._require()
        snapshot = self._snapshot(campaign)
        gate = gate_for(snapshot)
        if not gate["passed"]:
            raise ValueError(gate["reason"])
        coming = next_stage(campaign["stage"])
        if coming == "affine_mold":
            self._freeze_profile(campaign)
        campaign["stage"] = coming
        campaign["state"]["stage"] = coming
        self._save(campaign)
        self.storage.audit(campaign["id"], "campaign_advanced", {"stage": coming})
        return self.view()

    def observe_frame(self, frame_bgr: np.ndarray) -> dict[str, Any] | None:
        scene = self.camera_scene()
        if scene not in {"board", "undistort", "focus"}:
            with self._lock:
                self.live_detection = None
            return None
        campaign = self._current()
        pattern = tuple(campaign["state"]["pattern"]) if campaign else PATTERN
        detection = detect_chessboard(frame_bgr, pattern)  # type: ignore[arg-type]
        with self._lock:
            self.live_detection = detection
        if campaign is not None:
            self._update_focus(campaign, frame_bgr, detection)
        return detection

    def preview_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        scene = self.camera_scene()
        if scene == "mold":
            return None
        campaign = self._current()
        detection = self.live_detection
        if scene == "undistort" and campaign and campaign["state"].get("intrinsic"):
            intrinsic = campaign["state"]["intrinsic"]
            preview = preview_before_after(
                frame_bgr,
                np.asarray(intrinsic["camera_matrix"], dtype=np.float64),
                np.asarray(intrinsic["dist_coeffs"], dtype=np.float64),
            )
            return preview
        if detection is None:
            detection = {"found": False, "corners": []}
        pattern = tuple(campaign["state"]["pattern"]) if campaign else PATTERN
        annotated = draw_detection(frame_bgr, detection, pattern)  # type: ignore[arg-type]
        if scene == "focus":
            reading = self.focus_view()
            label = reading["status"]
            if reading["current"] is not None:
                label = f"nitidez {reading['current']:.0f}  max {reading['peak']:.0f}  {reading['status']}"
            cv2.putText(annotated, label, (16, annotated.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def decide_capture(self, decision: str, reason: str = "") -> dict[str, Any]:
        if decision not in {"accepted", "rejected", "review"}:
            raise ValueError("A decisão deve ser aceitar, rejeitar ou revisar")
        campaign = self._require()
        self._require_focus_held(campaign)
        if campaign["stage"] not in {"capture_intrinsic", "solve_extrinsic", "validate_planes"}:
            raise ValueError("Esta etapa não captura tabuleiro")
        with self._lock:
            detection = dict(self.live_detection) if self.live_detection else None
        if detection is None or not detection.get("found"):
            raise ValueError("Tabuleiro não detectado neste frame. Ajuste a posição e tente de novo.")
        plan_z = None
        if campaign["stage"] == "validate_planes":
            plan_z = float(campaign["state"].get("validation_plan_z") if campaign["state"].get("validation_plan_z") is not None else 0.0)
        self._store_capture(
            campaign,
            corners=detection["corners"],
            image_size=(int(detection["width"]), int(detection["height"])),
            decision=decision,
            reason=reason,
            cell=detection.get("cell"),
            plan_z=plan_z,
        )
        return self.view()

    def record_capture(
        self,
        corners: list[list[float]],
        image_size: tuple[int, int],
        decision: str = "accepted",
        reason: str = "",
        plan_z: float | None = None,
        cell: list[int] | None = None,
    ) -> dict[str, Any]:
        """Grava cantos já detectados. Os testes e o simulador usam o mesmo caminho da HMI."""
        campaign = self._require()
        self._require_focus_held(campaign)
        if campaign["stage"] not in {"capture_intrinsic", "solve_extrinsic", "validate_planes"}:
            raise ValueError("Esta etapa não captura tabuleiro")
        if plan_z is None and campaign["stage"] == "validate_planes":
            plan_z = float(campaign["state"].get("validation_plan_z") or 0.0)
        self._store_capture(
            campaign,
            corners=corners,
            image_size=image_size,
            decision=decision,
            reason=reason,
            cell=cell,
            plan_z=plan_z,
        )
        return self.view()

    def solve_intrinsic(self) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "review_reprojection")
        accepted = self._accepted(campaign, "capture_intrinsic")
        if len(accepted) < 3:
            raise ValueError("Não há imagens aceitas para o cálculo")
        size = campaign["state"].get("image_size_px")
        if not size:
            raise ValueError("As capturas não registraram o tamanho da imagem")
        solution = calibrate_intrinsic(
            [item["corners"] for item in accepted],
            (int(size[0]), int(size[1])),
            float(campaign["state"]["square_size_mm"]),
            tuple(campaign["state"]["pattern"]),  # type: ignore[arg-type]
        )
        for item, error in zip(accepted, solution["per_image_px"], strict=True):
            self.storage.update_campaign_capture(item["id"], reprojection_px=error)
        solution["capture_ids"] = [item["id"] for item in accepted]
        campaign["state"]["intrinsic"] = solution
        campaign["state"]["reprojection_reviewed"] = False
        self._save(campaign)
        return self.view()

    def exclude_capture(self, capture_id: str) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "review_reprojection")
        self.storage.update_campaign_capture(capture_id, decision="rejected", reason="Erro de reprojeção")
        campaign["state"]["intrinsic"] = None
        campaign["state"]["reprojection_reviewed"] = False
        self._save(campaign)
        return self.view()

    def confirm_reprojection(self) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "review_reprojection")
        campaign["state"]["reprojection_reviewed"] = True
        self._save(campaign)
        gate = gate_for(self._snapshot(campaign))
        if not gate["passed"] and "inspecionada" not in gate["reason"]:
            campaign["state"]["reprojection_reviewed"] = False
            self._save(campaign)
            raise ValueError(gate["reason"])
        return self.view()

    def confirm_undistort(self) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "confirm_undistort")
        campaign["state"]["undistort_confirmed"] = True
        self._save(campaign)
        return self.view()

    def record_frame_point(self, role: str, robot: dict[str, float] | None = None) -> dict[str, Any]:
        if role not in {"origin", "plus_x", "plus_y"}:
            raise ValueError("O ponto do referencial deve ser origem, +X ou +Y")
        campaign = self._require()
        self._require_stage(campaign, "teach_frame")
        pose = robot if robot is not None else self._robot_xyz()
        points = dict(campaign["state"].get("frame_points") or {})
        points[role] = {"x": pose["x"], "y": pose["y"], "z": pose["z"], "recorded_at": utc_now()}
        campaign["state"]["frame_points"] = points
        campaign["state"]["board_frame"] = self._try_board_frame(points)
        self._save(campaign)
        return self.view()

    def solve_pose(self) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "solve_extrinsic")
        accepted = self._accepted(campaign, "solve_extrinsic")
        if len(accepted) != 1:
            raise ValueError("Deixe exatamente uma imagem aceita desta pose")
        intrinsic = campaign["state"].get("intrinsic")
        frame = campaign["state"].get("board_frame")
        if not intrinsic or not frame:
            raise ValueError("A intrínseca e o referencial precisam estar prontos")
        solved = solve_extrinsic(
            accepted[0]["corners"],
            intrinsic["camera_matrix"],
            intrinsic["dist_coeffs"],
            float(campaign["state"]["square_size_mm"]),
            tuple(campaign["state"]["pattern"]),  # type: ignore[arg-type]
        )
        solved["capture_id"] = accepted[0]["id"]
        solved["board_to_robot"] = frame
        solved["robot_from_camera"] = robot_from_camera(solved["object_to_camera"], frame)
        campaign["state"]["extrinsic"] = solved
        campaign["state"]["direction_check"] = None
        campaign["state"]["direction"] = None
        self._save(campaign)
        return self.view()

    def record_direction_touch(self, robot: dict[str, float] | None = None) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "confirm_direction")
        extrinsic = campaign["state"].get("extrinsic")
        if not extrinsic:
            raise ValueError("Calcule a extrínseca antes de conferir o sentido")
        pose = robot if robot is not None else self._robot_xyz()
        check = direction_errors(
            [pose["x"], pose["y"], pose["z"]],
            extrinsic["board_to_robot"],
            float(campaign["state"]["square_size_mm"]),
        )
        check["tcp_mm"] = [pose["x"], pose["y"], pose["z"]]
        campaign["state"]["direction_check"] = check
        campaign["state"]["direction"] = None
        self._save(campaign)
        return self.view()

    def confirm_direction(self, sense: str) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "confirm_direction")
        if sense != "object_to_camera":
            raise ValueError("O sentido invertido não avança. Confirme tabuleiro → câmera.")
        check = campaign["state"].get("direction_check") or {}
        if "error_correct_mm" not in check:
            raise ValueError("Grave a pose do TCP no canto de conferência")
        if float(check["error_correct_mm"]) > float(check["error_inverted_mm"]):
            raise ValueError("O sentido invertido está mais perto do TCP. Reensine os eixos.")
        campaign["state"]["direction"] = "object_to_camera"
        self._save(campaign)
        return self.view()

    def select_validation_plane(self, plan_z: float) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "validate_planes")
        if not any(abs(float(plan_z) - plane) < 1e-6 for plane in VALIDATION_PLANES_MM):
            raise ValueError("A validação usa os planos 0, 200 e 400 mm")
        campaign["state"]["validation_plan_z"] = float(plan_z)
        self._save(campaign)
        return self.view()

    def open_affine_session(self, name: str | None = None) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "affine_mold")
        profile = campaign["state"].get("profile")
        if not profile or not campaign["state"].get("profile_sha256"):
            raise ValueError("O perfil óptico ainda não foi fechado")
        planes = [float(value) for value in self.config["calibration"]["default_planes_mm"]]
        session = self.capture.create_session(name or f"{campaign['name']} · molde", planes)
        optical = {
            "profile_id": campaign["id"],
            "profile_sha256": campaign["state"]["profile_sha256"],
            "input_space": "undistorted_px_K_rect",
            "stale": False,
            "intrinsic": {
                "camera_matrix": profile["intrinsic"]["camera_matrix"],
                "dist_coeffs": profile["intrinsic"]["dist_coeffs"],
                "rectified_camera_matrix": profile["intrinsic"]["rectified_camera_matrix"],
            },
        }
        self.storage.merge_session_config(session["id"], {"optical_profile": optical})
        self.storage.mark_profiles_stale(campaign["state"]["profile_sha256"])
        self.capture.activate_session(session["id"])
        campaign["state"]["affine_session_id"] = session["id"]
        self._save(campaign)
        return self.view()

    def apply_ransac(self) -> dict[str, Any]:
        campaign = self._require()
        self._require_stage(campaign, "analyze_residuals")
        diagnosis = self._diagnosis(campaign)
        if not diagnosis["ransac_available"]:
            raise ValueError("O RANSAC só aparece quando poucos pontos estão muito ruins")
        session_id = campaign["state"].get("affine_session_id")
        if not session_id:
            raise ValueError("Não há sessão de molde ligada a esta campanha")
        tolerance = float(self.config["calibration"]["xy_tolerance_mm"])
        offset = self.config["calibration"]["pick_offset_local_mm"]
        detail = self.capture.session_detail(session_id)
        profile = detail["config"].get("optical_profile")
        refit: list[dict[str, Any]] = []
        for plan in detail["plans"]:
            if not plan["evaluation"].get("ready"):
                continue
            kept = []
            for pair in detail["pairs"]:
                if float(pair["plan_z"]) != float(plan["z"]) or pair["role"] != "adjustment":
                    continue
                residual = self._pair_residual(plan["evaluation"], pair["id"])
                if residual is None or residual <= tolerance:
                    kept.append(pair)
            if len(kept) < 3:
                continue
            if profile and not profile.get("stale"):
                kept = undistort_pairs(kept, profile)
            model = fit_plan_calibration(float(plan["z"]), kept, offset)
            refit.append({"z": plan["z"], "kept": len(kept), "model": model.to_dict()})
        campaign["state"]["ransac"] = {"applied_at": utc_now(), "planes": refit}
        self._save(campaign)
        return self.view()

    def view(self) -> dict[str, Any] | None:
        campaign = self._current()
        if campaign is None:
            return None
        self._publish_simulation(campaign)
        snapshot = self._snapshot(campaign)
        gate = gate_for(snapshot)
        stage = campaign["stage"]
        index = STAGES.index(stage)
        accepted = [item for item in campaign["captures"] if item["decision"] == "accepted" and item["stage"] == "capture_intrinsic"]
        intrinsic = campaign["state"].get("intrinsic") or {}
        images = []
        for item in accepted:
            error = item.get("reprojection_px")
            images.append(
                {
                    "id": item["id"],
                    "error_px": error,
                    "suspect": error is None or float(error) > 1.0,
                }
            )
        with self._lock:
            live = dict(self.live_detection) if self.live_detection else None
        if live:
            live = {"found": bool(live.get("found")), "count": int(live.get("count") or 0), "cell": live.get("cell")}
        return {
            "id": campaign["id"],
            "name": campaign["name"],
            "stage": stage,
            "stage_index": index,
            "square_size_mm": campaign["state"]["square_size_mm"],
            "pattern": campaign["state"]["pattern"],
            "checks": campaign["state"]["checks"],
            "copy": STAGE_GUIDE[stage],
            "gate": gate,
            "stages": [
                {
                    "id": name,
                    "title": STAGE_GUIDE[name]["title"],
                    "state": "done" if position < index else "current" if position == index else "locked",
                }
                for position, name in enumerate(STAGES)
            ],
            "accepted_count": len(accepted),
            "coverage": coverage_view(campaign["captures"]),
            "detection": live,
            "reprojection": {
                "rms_px": intrinsic.get("rms_px"),
                "images": images,
                "reviewed": bool(campaign["state"].get("reprojection_reviewed")),
            },
            "frame_points": {
                role: bool((campaign["state"].get("frame_points") or {}).get(role))
                for role in ("origin", "plus_x", "plus_y")
            },
            "direction_check": campaign["state"].get("direction_check"),
            "direction": campaign["state"].get("direction"),
            "validation_plan_z": campaign["state"].get("validation_plan_z"),
            "validation": self._validation_summary(campaign),
            "affine_session_id": campaign["state"].get("affine_session_id"),
            "profile_sha256": campaign["state"].get("profile_sha256"),
            "diagnosis": self._diagnosis(campaign) if stage == "analyze_residuals" else None,
            "extrinsic_ready": bool(campaign["state"].get("extrinsic")),
            "image_size_px": campaign["state"].get("image_size_px"),
            "focus": self.focus_view(),
        }

    def _store_capture(
        self,
        campaign: dict[str, Any],
        corners: list[list[float]],
        image_size: tuple[int, int],
        decision: str,
        reason: str,
        cell: list[int] | None,
        plan_z: float | None,
    ) -> None:
        expected = int(campaign["state"]["pattern"][0]) * int(campaign["state"]["pattern"][1])
        if decision == "accepted" and len(corners) != expected:
            raise ValueError(f"A imagem precisa ter {expected} cantos internos")
        size = [int(image_size[0]), int(image_size[1])]
        if corners:
            cell = coverage_cell(np.asarray(corners, dtype=np.float64), size[0], size[1])
        known = campaign["state"].get("image_size_px")
        if decision == "accepted":
            if known and list(known) != size:
                raise ValueError("A resolução mudou. O perfil só vale para o raster da primeira captura.")
            campaign["state"]["image_size_px"] = size
        if campaign["stage"] == "solve_extrinsic" and decision == "accepted":
            for item in self._accepted(campaign, "solve_extrinsic"):
                self.storage.update_campaign_capture(item["id"], decision="rejected", reason="Substituída pela nova pose")
            campaign["state"]["extrinsic"] = None
        metrics = None
        if campaign["stage"] == "validate_planes" and decision == "accepted":
            metrics = self._metrics_for(campaign, corners, float(plan_z if plan_z is not None else 0.0))
        self.storage.add_campaign_capture(
            campaign["id"],
            {
                "stage": campaign["stage"],
                "decision": decision,
                "reason": reason,
                "corners": corners,
                "cell": cell,
                "plan_z": plan_z,
                "metrics": metrics,
            },
        )
        if campaign["stage"] == "capture_intrinsic":
            campaign["state"]["intrinsic"] = None
            campaign["state"]["reprojection_reviewed"] = False
        self._save(campaign)

    def _metrics_for(self, campaign: dict[str, Any], corners: list[list[float]], plan_z: float) -> dict[str, Any]:
        intrinsic = campaign["state"].get("intrinsic")
        extrinsic = campaign["state"].get("extrinsic")
        if not intrinsic or not extrinsic:
            raise ValueError("Valide só depois da intrínseca e da extrínseca")
        return validation_metrics(
            corners,
            extrinsic["board_to_robot"],
            extrinsic["robot_from_camera"],
            intrinsic["camera_matrix"],
            intrinsic["dist_coeffs"],
            float(campaign["state"]["square_size_mm"]),
            plan_z,
            tuple(campaign["state"]["pattern"]),  # type: ignore[arg-type]
        )

    def _freeze_profile(self, campaign: dict[str, Any]) -> None:
        state = campaign["state"]
        state["fingerprint"] = self._fingerprint(state)
        payload = {
            "schema_version": "optical-1",
            "campaign_id": campaign["id"],
            "image_size_px": state.get("image_size_px"),
            "pattern": state["pattern"],
            "square_size_mm": state["square_size_mm"],
            "fingerprint": state["fingerprint"],
            "intrinsic": {
                "model": state["intrinsic"]["model"],
                "camera_matrix": state["intrinsic"]["camera_matrix"],
                "dist_coeffs": state["intrinsic"]["dist_coeffs"],
                "rectified_camera_matrix": state["intrinsic"]["rectified_camera_matrix"],
                "rms_px": state["intrinsic"]["rms_px"],
                "valid_roi_px": state["intrinsic"]["valid_roi_px"],
            },
            "extrinsic": {
                "object_to_camera": state["extrinsic"]["object_to_camera"],
                "camera_to_object": state["extrinsic"]["camera_to_object"],
                "board_to_robot": state["extrinsic"]["board_to_robot"],
                "robot_from_camera": state["extrinsic"]["robot_from_camera"],
                "direction": "object_to_camera",
            },
            "affine_included": False,
        }
        digest = canonical_hash(payload)
        payload["profile_sha256"] = digest
        directory = self.project_root / "data" / "campaigns" / campaign["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "profile.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        state["profile"] = payload
        state["profile_sha256"] = digest
        state["fingerprint"] = payload["fingerprint"]
        self.storage.mark_profiles_stale(digest)

    def _fingerprint(self, state: dict[str, Any]) -> dict[str, Any]:
        camera = self.config.get("camera", {})
        lock = state.get("focus_lock") or {}
        return {
            "image_size_px": state.get("image_size_px"),
            "camera_provider": camera.get("provider"),
            "max_dimension": camera.get("max_dimension"),
            "focus_sharpness_peak": lock.get("peak"),
            "note": "K e D valem somente para este foco e para este raster, depois do downscale de produção",
        }

    def _publish_simulation(self, campaign: dict[str, Any]) -> None:
        stage = campaign["stage"]
        scene = "mold"
        if stage == "fix_hardware":
            scene = "focus"
        elif stage in BOARD_STAGES and stage != "confirm_undistort":
            scene = "board"
        elif stage == "confirm_undistort":
            scene = "undistort"
        override = self._simulation_pose(campaign)
        with self._lock:
            self._scene = scene
            self._pose_override = override

    def _simulation_pose(self, campaign: dict[str, Any]) -> tuple[float, float, float] | None:
        stage = campaign["stage"]
        points = campaign["state"].get("frame_points") or {}
        if stage == "teach_frame":
            if not points.get("origin"):
                return (100.0, 200.0, 0.0)
            if not points.get("plus_x"):
                return (400.0, 200.0, 0.0)
            if not points.get("plus_y"):
                return (100.0, 450.0, 0.0)
            return None
        if stage == "confirm_direction":
            frame = campaign["state"].get("board_frame")
            if not frame:
                return None
            predicted = direction_errors([0, 0, 0], frame, float(campaign["state"]["square_size_mm"]))
            point = predicted["predicted_mm"]
            return (float(point[0]), float(point[1]), float(point[2]))
        return None

    def _diagnosis(self, campaign: dict[str, Any]) -> dict[str, Any]:
        session_id = campaign["state"].get("affine_session_id")
        if not session_id:
            return {"findings": ["Abra a coleta do molde e grave pontos antes de ler os resíduos."], "outlier_count": 0, "ransac_available": False}
        try:
            detail = self.capture.session_detail(session_id)
        except KeyError:
            return {"findings": ["A sessão do molde não foi encontrada."], "outlier_count": 0, "ransac_available": False}
        samples: list[dict[str, float]] = []
        width = 1.0
        height = 1.0
        for plan in detail["plans"]:
            evaluation = plan["evaluation"]
            if not evaluation.get("ready"):
                continue
            for pair in detail["pairs"]:
                if float(pair["plan_z"]) != float(plan["z"]):
                    continue
                residual = self._pair_residual(evaluation, pair["id"])
                if residual is None:
                    continue
                vision = pair["vision"]
                width = float(vision.get("frame_width") or width)
                height = float(vision.get("frame_height") or height)
                samples.append(
                    {
                        "error_mm": float(residual),
                        "u_norm": float(vision["x"]) / max(width, 1.0),
                        "v_norm": float(vision["y"]) / max(height, 1.0),
                        "z": float(plan["z"]),
                    }
                )
        diagnosis = diagnose_residuals(samples, float(self.config["calibration"]["xy_tolerance_mm"]))
        if campaign["state"].get("ransac"):
            diagnosis["findings"] = [*diagnosis["findings"], "RANSAC registrado. Os pontos originais permanecem na trilha."]
            diagnosis["ransac_available"] = False
        return diagnosis

    @staticmethod
    def _pair_residual(evaluation: dict[str, Any], pair_id: str) -> float | None:
        for item in evaluation.get("metrics", {}).get("pairs", []):
            if item.get("pair_id") == pair_id:
                return float(item["residual_xy_mm"])
        return None

    def _validation_summary(self, campaign: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for plane in VALIDATION_PLANES_MM:
            rows = [
                item for item in campaign["captures"]
                if item["stage"] == "validate_planes" and item["decision"] == "accepted" and item.get("metrics")
                and abs(float(item.get("plan_z") or -1) - plane) < 1e-6
            ]
            if not rows:
                summary[f"{plane:.0f}"] = None
                continue
            mm = rows[-1]["metrics"]["millimetres"]
            summary[f"{plane:.0f}"] = {"mae": mm["mae"], "rmse": mm["rmse"], "max": mm["max"], "count": len(rows)}
        return summary

    def _snapshot(self, campaign: dict[str, Any]) -> dict[str, Any]:
        affine_ready = False
        session_id = campaign["state"].get("affine_session_id")
        if session_id:
            try:
                detail = self.capture.session_detail(session_id)
                affine_ready = any(plan["evaluation"].get("ready") for plan in detail["plans"])
            except KeyError:
                affine_ready = False
        return {
            "stage": campaign["stage"],
            "checks": campaign["state"].get("checks") or {},
            "square_size_mm": campaign["state"].get("square_size_mm"),
            "captures": campaign["captures"],
            "intrinsic": campaign["state"].get("intrinsic"),
            "reprojection_reviewed": campaign["state"].get("reprojection_reviewed"),
            "undistort_confirmed": campaign["state"].get("undistort_confirmed"),
            "frame_points": campaign["state"].get("frame_points") or {},
            "board_frame": campaign["state"].get("board_frame"),
            "extrinsic": campaign["state"].get("extrinsic"),
            "direction": campaign["state"].get("direction"),
            "direction_check": campaign["state"].get("direction_check"),
            "affine_session_id": session_id,
            "affine_ready": affine_ready,
            "focus_lock": bool(campaign["state"].get("focus_lock")),
            "focus_drift": bool(campaign["state"].get("focus_drift")),
        }

    def _save(self, campaign: dict[str, Any]) -> None:
        state = dict(campaign["state"])
        state["stage"] = campaign["stage"]
        self.storage.save_campaign_state(campaign["id"], campaign["stage"], state)
        self._publish_simulation(campaign)

    def _current(self) -> dict[str, Any] | None:
        with self._lock:
            campaign_id = self.active_id
        if campaign_id is None:
            return None
        return self.storage.get_campaign(campaign_id)

    def _require(self) -> dict[str, Any]:
        campaign = self._current()
        if campaign is None:
            raise ValueError("Abra uma campanha de calibração completa")
        return campaign

    @staticmethod
    def _require_stage(campaign: dict[str, Any], stage: str) -> None:
        if campaign["stage"] != stage:
            title = STAGE_GUIDE.get(campaign["stage"], {}).get("title", campaign["stage"])
            raise ValueError(f"Esta ação pertence a outra etapa. A etapa atual é: {title}")

    @staticmethod
    def _accepted(campaign: dict[str, Any], stage: str) -> list[dict[str, Any]]:
        return [
            item for item in campaign["captures"]
            if item["stage"] == stage and item["decision"] == "accepted"
        ]

    def _robot_xyz(self) -> dict[str, float]:
        robot = self.capture.robot_supplier()
        if robot is None or not robot.fresh:
            raise ValueError("Pose do robô indisponível")
        return {"x": float(robot.x), "y": float(robot.y), "z": float(robot.z)}

    @staticmethod
    def _try_board_frame(points: dict[str, Any]) -> dict[str, Any] | None:
        if not all(points.get(role) for role in ("origin", "plus_x", "plus_y")):
            return None
        try:
            return board_frame(
                [points["origin"]["x"], points["origin"]["y"], points["origin"]["z"]],
                [points["plus_x"]["x"], points["plus_x"]["y"], points["plus_x"]["z"]],
                [points["plus_y"]["x"], points["plus_y"]["y"], points["plus_y"]["z"]],
            )
        except ValueError:
            return None

    def _reset_focus_tracker(self) -> None:
        self._focus_samples: list[float] = []
        self._focus_peak = 0.0
        self._focus_below = 0
        self._focus_board = False
        self._focus_current: float | None = None

    def _focus_is_stable(self) -> bool:
        locked = None
        assessed = assess_focus(self._focus_samples, self._focus_peak, locked)
        return bool(assessed["stable"] and self._focus_board)

    def _update_focus(self, campaign: dict[str, Any], frame_bgr: np.ndarray, detection: dict[str, Any]) -> None:
        if not detection.get("found"):
            self._focus_board = False
            self._focus_below = 0
            return
        score = sharpness_score(frame_bgr, detection.get("corners"))
        self._focus_samples = [*self._focus_samples, score][-12:]
        lock = campaign["state"].get("focus_lock") or {}
        assessed = assess_focus(self._focus_samples, self._focus_peak, lock.get("peak"))
        self._focus_peak = float(assessed["peak"])
        self._focus_current = assessed["current"]
        self._focus_board = True
        if assessed["soft"]:
            self._focus_below += 1
        else:
            self._focus_below = 0
        if self._focus_below >= 8 and lock.get("peak") and not campaign["state"].get("focus_drift"):
            campaign["state"]["focus_drift"] = True
            self._save(campaign)

    def focus_view(self) -> dict[str, Any]:
        campaign = self._current()
        lock = (campaign or {}).get("state", {}).get("focus_lock") if campaign else None
        drift = bool(campaign and campaign["state"].get("focus_drift"))
        assessed = assess_focus(self._focus_samples, self._focus_peak, (lock or {}).get("peak"))
        locked_peak = None if not lock else lock.get("peak")
        if drift:
            status = "O foco caiu. Recomece a campanha."
        elif locked_peak:
            status = "Foco travado."
        elif not self._focus_board:
            status = "Mostre o tabuleiro para medir a nitidez."
        elif assessed["stable"]:
            status = "Nitidez estável no máximo. Trave o anel e marque o foco."
        elif assessed["peak"] > 1.0 and assessed["current"] is not None and assessed["current"] < 0.90 * assessed["peak"]:
            status = "Nitidez abaixo do máximo. Gire o anel devagar."
        else:
            status = "Gire o anel até a nitidez parar de subir."
        return {
            "current": assessed["current"],
            "peak": locked_peak if locked_peak else assessed["peak"],
            "stable": bool(assessed["stable"] and self._focus_board and not drift),
            "board_seen": self._focus_board,
            "locked_peak": None if not lock else lock.get("peak"),
            "drift": drift,
            "status": status,
        }

    def _require_focus_held(self, campaign: dict[str, Any]) -> None:
        if campaign["state"].get("focus_drift"):
            raise ValueError("O foco caiu em relação ao valor travado. Recomece a campanha.")

    def encode_preview(self, frame_bgr: np.ndarray) -> bytes | None:
        preview = self.preview_frame(frame_bgr)
        if preview is None:
            return None
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            return None
        return encoded.tobytes()
