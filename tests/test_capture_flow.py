from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sincro_robo.config import load_config
from sincro_robo.domain import RobotPoseSnapshot, VisionObservation, utc_now
from sincro_robo.services.capture import CaptureController
from sincro_robo.storage import Storage


def valid_vision(x: float = 480.0, y: float = 270.0) -> VisionObservation:
    return VisionObservation(
        timestamp=utc_now(), x=x, y=y, angle_deg=25.0, confidence=0.99,
        axis_quality=4.2, mask_area_ratio=0.12, mask_cut=False,
        instance_count=1, stable=True, sigma_x=0.2, sigma_y=0.2,
        sigma_angle_deg=0.1, frame_width=960, frame_height=540,
        gates={"single_instance": True, "confidence": True, "mask_not_cut": True,
               "axis_quality": True, "mask_area": True},
    )


def valid_robot() -> RobotPoseSnapshot:
    return RobotPoseSnapshot(utc_now(), 320.0, 210.0, 0.0, 180.0, 0.0, 37.0, True)


@pytest.fixture
def controller(tmp_path: Path) -> CaptureController:
    config = load_config("config/simulator.json")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    storage = Storage(str(tmp_path / "test.sqlite3"))
    return CaptureController(storage, config, valid_vision, valid_robot, tmp_path)


def test_candidate_is_not_persisted_until_human_confirmation(controller: CaptureController) -> None:
    session = controller.create_session("Teste confirmação", [0.0])
    controller.activate_plan(session["id"], 0.0)
    candidate = controller.capture()
    assert candidate.point_index == 1
    assert controller.storage.list_pairs(session["id"]) == []
    controller.decide(False)
    assert controller.storage.list_pairs(session["id"]) == []


def test_confirm_persists_exactly_one_pair(controller: CaptureController) -> None:
    session = controller.create_session("Teste gravação", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    result = controller.decide(True)
    assert result["saved"]
    pairs = controller.storage.list_pairs(session["id"])
    assert len(pairs) == 1
    assert pairs[0]["robot"]["rx"] == pytest.approx(180.0)
    assert pairs[0]["robot"]["ry"] == pytest.approx(0.0)


def test_duplicate_pair_is_rejected(controller: CaptureController) -> None:
    session = controller.create_session("Teste duplicidade", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    controller.decide(True)
    with pytest.raises(ValueError, match="duplicado"):
        controller.capture()


def test_unstable_vision_cannot_be_frozen(controller: CaptureController) -> None:
    unstable = valid_vision()
    object.__setattr__(unstable, "stable", False)
    controller.vision_supplier = lambda: unstable
    session = controller.create_session("Teste gate", [0.0])
    controller.activate_plan(session["id"], 0.0)
    with pytest.raises(ValueError, match="estável"):
        controller.capture()


def test_region_gate_rejects_point_outside_suggestion(controller: CaptureController) -> None:
    controller.vision_supplier = lambda: valid_vision(850.0, 480.0)
    session = controller.create_session("Teste região", [0.0])
    controller.activate_plan(session["id"], 0.0)
    assert not controller.capture_readiness()["region"]
    with pytest.raises(ValueError, match="região útil"):
        controller.capture()


def test_plan_gate_rejects_wrong_robot_z(controller: CaptureController) -> None:
    controller.robot_supplier = lambda: replace(valid_robot(), z=200.0)
    session = controller.create_session("Teste plano", [0.0])
    controller.activate_plan(session["id"], 0.0)
    assert not controller.capture_readiness()["plan_z"]
    with pytest.raises(ValueError, match="fora do plano"):
        controller.capture()
