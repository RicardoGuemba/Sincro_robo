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


def test_simulator_session_records_identity_pixel_reference(controller: CaptureController) -> None:
    session = controller.create_session("Simulador", [0.0, 200.0, 400.0])
    reference = session["config"]["pixel_reference"]
    assert reference["source_width"] == 960
    assert reference["source_height"] == 540
    assert reference["destination_width"] == 960
    assert reference["destination_height"] == 540
    assert reference["scale_x"] == pytest.approx(1.0)
    assert reference["scale_y"] == pytest.approx(1.0)


def test_duplicate_pair_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    session = controller.create_session("Teste duplicidade", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    controller.decide(True)
    readiness = controller.capture_readiness()
    assert not readiness["not_duplicate"]
    assert readiness["ready"]
    candidate = controller.capture()
    assert candidate.point_index == 2


def test_unstable_vision_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    unstable = valid_vision()
    object.__setattr__(unstable, "stable", False)
    controller.vision_supplier = lambda: unstable
    session = controller.create_session("Teste gate", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert not readiness["stable"]
    assert readiness["ready"]
    candidate = controller.capture()
    assert candidate.vision.stable is False


def test_region_gate_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    controller.vision_supplier = lambda: valid_vision(850.0, 480.0)
    session = controller.create_session("Teste região", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert not readiness["region"]
    assert readiness["ready"]
    candidate = controller.capture()
    assert candidate.vision.x == pytest.approx(850.0)


def test_plan_gate_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    controller.robot_supplier = lambda: replace(valid_robot(), z=200.0)
    session = controller.create_session("Teste plano", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert not readiness["plan_z"]
    assert readiness["ready"]
    candidate = controller.capture()
    assert candidate.robot.z == pytest.approx(200.0)


def test_operator_can_capture_with_all_quality_gates_red(controller: CaptureController) -> None:
    vision = valid_vision(850.0, 480.0)
    object.__setattr__(vision, "stable", False)
    object.__setattr__(vision, "gates", {
        "single_instance": False,
        "confidence": False,
        "mask_not_cut": False,
        "axis_quality": False,
        "mask_area": False,
    })
    controller.vision_supplier = lambda: vision
    controller.robot_supplier = lambda: replace(valid_robot(), z=200.0, fresh=False)
    session = controller.create_session("Captura livre", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert readiness["ready"]
    assert not readiness["single_instance"]
    assert not readiness["stable"]
    assert not readiness["pose"]
    assert not readiness["region"]
    assert not readiness["plan_z"]
    candidate = controller.capture()
    assert candidate.session_id == session["id"]
