from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sincro_robo.adapters.segmenter import SegmentationResult
from sincro_robo.config import load_config
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
