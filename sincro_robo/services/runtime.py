from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2

from ..adapters.camera import StApiCamera, SyntheticCamera
from ..adapters.plc import CipPoseReader, SyntheticPoseReader
from ..adapters.segmenter import RFDetrSegmenter, SyntheticSegmenter, annotate_frame
from ..domain import RobotPoseSnapshot, VisionObservation, utc_now
from ..geometry import MoldPoseEstimator, VisionStabilityTracker, build_observation
from ..storage import Storage
from .capture import CaptureController


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.vision: VisionObservation | None = None
        self.robot: RobotPoseSnapshot | None = None
        self.frame_jpeg: bytes | None = None
        self.camera_status = "starting"
        self.model_status = "starting"
        self.plc_status = "starting"
        self.camera_error: str | None = None
        self.model_error: str | None = None
        self.plc_error: str | None = None
        self.updated_at = utc_now()

    def get_vision(self) -> VisionObservation | None:
        with self._lock:
            return self.vision

    def get_robot(self) -> RobotPoseSnapshot | None:
        with self._lock:
            return self.robot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "vision": self.vision.to_dict() if self.vision else None,
                "robot": self.robot.to_dict() if self.robot else None,
                "hardware": {
                    "camera": {"status": self.camera_status, "error": self.camera_error},
                    "model": {"status": self.model_status, "error": self.model_error},
                    "plc": {"status": self.plc_status, "error": self.plc_error},
                },
                "updated_at": self.updated_at,
            }


class ApplicationRuntime:
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.state = SharedState()
        self.storage = Storage(config["app"]["database"])
        self.controller = CaptureController(
            self.storage,
            config,
            self.state.get_vision,
            self.state.get_robot,
            project_root,
        )
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _make_camera(self) -> Any:
        settings = self.config["camera"]
        if settings["provider"] == "stapi":
            return StApiCamera(settings["device_index"], settings["fetch_timeout_ms"])
        return SyntheticCamera(settings["width"], settings["height"])

    def _make_segmenter(self) -> Any:
        settings = self.config["model"]
        if settings["provider"] == "rfdetr":
            return RFDetrSegmenter(
                settings["checkpoint"],
                settings["threshold"],
                settings["resolution"],
                settings["class_name"],
                settings["device"],
            )
        return SyntheticSegmenter()

    def _make_plc(self) -> Any:
        settings = self.config["plc"]
        if settings["provider"] == "cip":
            return CipPoseReader(
                settings["ip"],
                settings["pose_tag"],
                settings["connection_timeout_s"],
                settings["angle_unit"],
            )
        offset = tuple(float(value) for value in self.config["calibration"]["pick_offset_local_mm"])
        return SyntheticPoseReader(
            self.state.get_vision,
            lambda: self.controller.active_plan_z,
            offset,
        )

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._vision_loop, name="camera-owner", daemon=True),
            threading.Thread(target=self._plc_loop, name="cip-owner", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _vision_loop(self) -> None:
        camera = self._make_camera()
        segmenter = self._make_segmenter()
        quality = self.config["vision_quality"]
        estimator = MoldPoseEstimator(quality["border_margin_px"])
        stability = VisionStabilityTracker(
            quality["stability_samples"],
            quality["max_jitter_x_px"],
            quality["max_jitter_y_px"],
            quality["max_jitter_angle_deg"],
        )
        frame_period = 1.0 / max(0.5, float(self.config["camera"]["target_fps"]))
        try:
            camera.open()
            with self.state._lock:
                self.state.camera_status = "online"
                self.state.camera_error = None
            segmenter.load()
            with self.state._lock:
                self.state.model_status = "online"
                self.state.model_error = None
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    frame = camera.grab()
                    with self.state._lock:
                        self.state.camera_status = "online"
                        self.state.camera_error = None
                except Exception as error:
                    stability.clear()
                    with self.state._lock:
                        self.state.vision = None
                        self.state.camera_status = "error"
                        self.state.camera_error = str(error)
                        self.state.updated_at = utc_now()
                    self._stop.wait(frame_period)
                    continue
                try:
                    result = segmenter.predict(frame)
                    observation = None
                    display_mask = None
                    if result.masks:
                        best_index = max(range(len(result.masks)), key=lambda index: result.confidences[index])
                        display_mask = result.masks[best_index]
                        pose = estimator.estimate(display_mask)
                        observation = build_observation(
                            pose,
                            result.confidences[best_index],
                            len(result.masks),
                            frame.shape,
                            quality,
                            self.config["pixel_reference"],
                        )
                        observation = stability.update(observation)
                    else:
                        stability.clear()
                    overlay_x = overlay_y = None
                    if observation is not None:
                        overlay_x, overlay_y = observation.overlay_xy()
                    annotated = annotate_frame(
                        frame,
                        display_mask,
                        overlay_x,
                        overlay_y,
                        observation.angle_deg if observation else None,
                        observation.stable if observation else False,
                    )
                    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 86])
                    with self.state._lock:
                        self.state.vision = observation
                        self.state.frame_jpeg = encoded.tobytes() if ok else None
                        self.state.camera_status = "online"
                        self.state.model_status = "online"
                        self.state.camera_error = None
                        self.state.model_error = None
                        self.state.updated_at = utc_now()
                except Exception as error:
                    stability.clear()
                    with self.state._lock:
                        self.state.vision = None
                        self.state.model_status = "error"
                        self.state.model_error = str(error)
                        self.state.updated_at = utc_now()
                remaining = frame_period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except Exception as error:
            with self.state._lock:
                if self.state.camera_status != "online":
                    self.state.camera_status = "error"
                    self.state.camera_error = str(error)
                else:
                    self.state.model_status = "error"
                    self.state.model_error = str(error)
        finally:
            try:
                camera.close()
            except Exception:
                pass
            with self.state._lock:
                if not self._stop.is_set():
                    self.state.camera_status = "error"
                else:
                    self.state.camera_status = "offline"
                    self.state.model_status = "offline"

    def _plc_loop(self) -> None:
        poll_interval = float(self.config["plc"]["poll_interval_s"])
        try:
            reader = self._make_plc()
        except Exception as error:
            with self.state._lock:
                self.state.robot = None
                self.state.plc_status = "error"
                self.state.plc_error = str(error)
                self.state.updated_at = utc_now()
            return
        connected = False
        while not self._stop.is_set():
            try:
                if not connected:
                    reader.connect()
                    connected = True
                pose = reader.read_pose()
                with self.state._lock:
                    self.state.robot = pose
                    self.state.plc_status = "online"
                    self.state.plc_error = None
                    self.state.updated_at = utc_now()
                self._stop.wait(poll_interval)
            except Exception as error:
                connected = False
                try:
                    reader.close()
                except Exception:
                    pass
                with self.state._lock:
                    self.state.robot = None
                    self.state.plc_status = "error"
                    self.state.plc_error = str(error)
                    self.state.updated_at = utc_now()
                self._stop.wait(min(3.0, max(0.5, poll_interval * 5)))
        try:
            reader.close()
        except Exception:
            pass
        with self.state._lock:
            self.state.plc_status = "offline"

    def snapshot(self) -> dict[str, Any]:
        payload = self.state.snapshot()
        payload["mode"] = {
            "camera": self.config["camera"]["provider"],
            "model": self.config["model"]["provider"],
            "plc": self.config["plc"]["provider"],
        }
        payload["candidate"] = self.controller.candidate.to_dict() if self.controller.candidate else None
        payload["capture_readiness"] = self.controller.capture_readiness()
        payload["active_session_id"] = self.controller.active_session_id
        payload["active_plan_z"] = self.controller.active_plan_z
        if self.controller.active_session_id:
            payload["session"] = self.controller.session_detail(self.controller.active_session_id)
        else:
            payload["session"] = None
        return payload
