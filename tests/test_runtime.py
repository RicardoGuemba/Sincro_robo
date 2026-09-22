from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from sincro_robo.adapters.segmenter import SegmentationResult
from sincro_robo.config import load_config
from sincro_robo.domain import RobotPoseSnapshot, utc_now
from sincro_robo.heartbeat import CIP_ECHO_ALARM
from sincro_robo.services.runtime import ApplicationRuntime


def test_vision_loop_loads_model_before_opening_camera(tmp_path: Path) -> None:
    order: list[str] = []

    class FakeCamera:
        def open(self) -> None:
            order.append("open")

        def grab(self) -> np.ndarray:
            order.append("grab")
            return np.zeros((32, 32, 3), dtype=np.uint8)

        def close(self) -> None:
            order.append("close")

    class FakeSegmenter:
        def load(self) -> None:
            order.append("load")

        def predict(self, frame: np.ndarray) -> SegmentationResult:
            order.append("predict")
            return SegmentationResult((), (), ())

    config = load_config("config/simulator.json")
    config["app"]["database"] = str(tmp_path / "runtime.sqlite3")
    config["camera"]["target_fps"] = 30.0
    runtime = ApplicationRuntime(config, tmp_path)
    runtime._make_camera = lambda: FakeCamera()  # type: ignore[method-assign]
    runtime._make_segmenter = lambda: FakeSegmenter()  # type: ignore[method-assign]
    runtime.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and runtime.state.frame_jpeg is None:
            time.sleep(0.02)
        assert runtime.state.frame_jpeg is not None
        assert order[:2] == ["load", "open"]
        assert "grab" in order
        assert "predict" in order
    finally:
        runtime.stop()


class FakeCipReader:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.written = False
        self.echo = False
        self.follow_echo = True
        self.fail = False
        self.x = 11.0
        self.connects = 0
        self.errors = 0

    def connect(self) -> None:
        with self._lock:
            if self.fail:
                self.errors += 1
                raise RuntimeError("connect fail")
            self.connects += 1

    def close(self) -> None:
        return None

    def write_bool(self, tag: str, value: bool) -> None:
        with self._lock:
            if self.fail:
                self.errors += 1
                raise RuntimeError("write fail")
            self.written = bool(value)
            if self.follow_echo:
                self.echo = self.written

    def read_bool(self, tag: str) -> bool:
        with self._lock:
            if self.fail:
                self.errors += 1
                raise RuntimeError("read fail")
            return self.echo

    def read_pose(self) -> RobotPoseSnapshot:
        with self._lock:
            if self.fail:
                self.errors += 1
                raise RuntimeError("pose fail")
            return RobotPoseSnapshot(utc_now(), self.x, 22.0, 200.0, 180.0, 0.0, 10.0, True)


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_cip_loop_blocks_until_echo_edge_then_keeps_xyz_on_loss(tmp_path: Path) -> None:
    config = load_config("config/simulator.json")
    config["app"]["database"] = str(tmp_path / "runtime.sqlite3")
    config["plc"]["provider"] = "cip"
    config["plc"]["poll_interval_s"] = 0.02
    config["plc"]["heartbeat_interval_s"] = 0.05
    config["plc"]["heartbeat_lost_after_s"] = 0.2
    reader = FakeCipReader()
    reader.follow_echo = False
    runtime = ApplicationRuntime(config, tmp_path)
    runtime._make_plc = lambda: reader  # type: ignore[method-assign]
    runtime.start()
    try:
        assert _wait_until(lambda: runtime.state.robot is not None)
        first = runtime.state.snapshot()
        assert first["robot"]["x"] == 11.0
        assert first["robot"]["fresh"] is False
        assert first["hardware"]["plc"]["status"] == "error"
        assert first["hardware"]["plc"]["error"] == CIP_ECHO_ALARM
        assert first["hardware"]["plc"]["echo"] is None

        reader.follow_echo = True
        assert _wait_until(lambda: runtime.state.robot is not None and runtime.state.robot.fresh)
        live = runtime.state.snapshot()
        assert live["hardware"]["plc"]["status"] == "online"
        assert live["hardware"]["plc"]["error"] is None
        assert isinstance(live["hardware"]["plc"]["echo"], bool)
        first_echo = live["hardware"]["plc"]["echo"]
        assert _wait_until(lambda: runtime.state.snapshot()["hardware"]["plc"].get("echo") is not first_echo)

        reader.x = 99.0
        reader.follow_echo = False
        assert _wait_until(
            lambda: runtime.state.robot is not None
            and runtime.state.robot.fresh is False
            and runtime.state.robot.x == 99.0
        )
        stale = runtime.state.snapshot()
        assert stale["hardware"]["plc"]["status"] == "error"
        assert stale["hardware"]["plc"]["error"] == CIP_ECHO_ALARM
        assert stale["hardware"]["plc"]["echo"] is None
        assert stale["robot"]["x"] == 99.0

        reader.fail = True
        assert _wait_until(lambda: reader.errors >= 1)
        frozen = runtime.state.snapshot()
        assert frozen["robot"]["x"] == 99.0
        assert frozen["robot"]["fresh"] is False
        assert frozen["hardware"]["plc"]["error"] == CIP_ECHO_ALARM

        reader.fail = False
        reader.follow_echo = True
        assert _wait_until(lambda: runtime.state.robot is not None and runtime.state.robot.fresh)
        recovered = runtime.state.snapshot()
        assert recovered["hardware"]["plc"]["status"] == "online"
        assert recovered["robot"]["fresh"] is True
        assert reader.connects >= 2
    finally:
        runtime.stop()
