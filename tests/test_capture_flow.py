from __future__ import annotations

import math
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


def valid_robot(x: float = 320.0, y: float = 210.0, z: float = 0.0, rz: float = 37.0) -> RobotPoseSnapshot:
    return RobotPoseSnapshot(utc_now(), x, y, z, 180.0, 0.0, rz, True)


def register_point(controller: CaptureController) -> dict:
    frozen = controller.capture()
    assert frozen["saved"] is False
    assert frozen["step"] == "vision_frozen"
    saved = controller.capture()
    assert saved["saved"] is True
    return saved


@pytest.fixture
def controller(tmp_path: Path) -> CaptureController:
    config = load_config("config/simulator.json")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    storage = Storage(str(tmp_path / "test.sqlite3"))
    return CaptureController(storage, config, valid_vision, valid_robot, tmp_path)


def test_vision_freeze_is_not_persisted_until_robot_capture(controller: CaptureController) -> None:
    session = controller.create_session("Teste duas etapas", [0.0])
    controller.activate_plan(session["id"], 0.0)
    frozen = controller.capture()
    assert frozen["saved"] is False
    assert controller.storage.list_pairs(session["id"]) == []
    controller.decide(False)
    assert controller.storage.list_pairs(session["id"]) == []
    assert controller.frozen_vision is None


def test_robot_capture_persists_exactly_one_pair(controller: CaptureController) -> None:
    session = controller.create_session("Teste gravação", [0.0])
    controller.activate_plan(session["id"], 0.0)
    result = register_point(controller)
    assert result["feedback"]["suggestion_kind"] == "collect"
    pairs = controller.storage.list_pairs(session["id"])
    assert len(pairs) == 1
    assert pairs[0]["robot"]["rx"] == pytest.approx(180.0)
    assert pairs[0]["robot"]["ry"] == pytest.approx(0.0)


def test_vision_step_does_not_require_robot(controller: CaptureController) -> None:
    controller.robot_supplier = lambda: None
    session = controller.create_session("Sem pose no 1/2", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert readiness["step"] == "vision"
    assert readiness["ready"]
    frozen = controller.capture()
    assert frozen["saved"] is False
    assert controller.capture_readiness()["ready"] is False


def test_robot_step_keeps_frozen_vision_when_live_vision_vanishes(controller: CaptureController) -> None:
    session = controller.create_session("Robô cobre a peça", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    controller.vision_supplier = lambda: None
    readiness = controller.capture_readiness()
    assert readiness["step"] == "robot"
    assert readiness["ready"]
    saved = controller.capture()
    assert saved["saved"]
    assert saved["pair"]["vision"]["x"] == pytest.approx(480.0)


def test_simulator_session_records_identity_pixel_reference(controller: CaptureController) -> None:
    session = controller.create_session("Simulador", [0.0, 200.0, 400.0])
    reference = session["config"]["pixel_reference"]
    assert reference["source_width"] == 960
    assert reference["source_height"] == 720
    assert reference["destination_width"] == 960
    assert reference["destination_height"] == 720
    assert reference["scale_x"] == pytest.approx(1.0)
    assert reference["scale_y"] == pytest.approx(1.0)


def test_duplicate_pair_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    session = controller.create_session("Teste duplicidade", [0.0])
    controller.activate_plan(session["id"], 0.0)
    register_point(controller)
    readiness = controller.capture_readiness()
    assert not readiness["not_duplicate"]
    assert readiness["ready"]
    frozen = controller.capture()
    assert frozen["frozen"]["point_index"] == 2


def test_unstable_vision_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    unstable = valid_vision()
    object.__setattr__(unstable, "stable", False)
    controller.vision_supplier = lambda: unstable
    session = controller.create_session("Teste gate", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert not readiness["stable"]
    assert readiness["ready"]
    frozen = controller.capture()
    assert frozen["frozen"]["vision"]["stable"] is False


def test_region_gate_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    controller.vision_supplier = lambda: valid_vision(850.0, 480.0)
    session = controller.create_session("Teste região", [0.0])
    controller.activate_plan(session["id"], 0.0)
    readiness = controller.capture_readiness()
    assert not readiness["region"]
    assert readiness["ready"]
    frozen = controller.capture()
    assert frozen["frozen"]["vision"]["x"] == pytest.approx(850.0)


def test_plan_gate_is_advisory_and_still_capturable(controller: CaptureController) -> None:
    controller.robot_supplier = lambda: replace(valid_robot(), z=200.0)
    session = controller.create_session("Teste plano", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    readiness = controller.capture_readiness()
    assert not readiness["plan_z"]
    assert readiness["ready"]
    saved = controller.capture()
    assert saved["pair"]["robot"]["z"] == pytest.approx(200.0)


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
    assert readiness["step"] == "vision"
    assert not readiness["single_instance"]
    assert not readiness["stable"]
    assert not readiness["pose"]
    assert not readiness["region"]
    assert not readiness["plan_z"]
    frozen = controller.capture()
    assert frozen["frozen"]["session_id"] == session["id"]
    robot_ready = controller.capture_readiness()
    assert robot_ready["step"] == "robot"
    assert not robot_ready["ready"]
    assert not robot_ready["pose"]
    with pytest.raises(ValueError, match="CIP SEM ECO"):
        controller.capture()
    assert controller.frozen_vision is not None


def test_robot_capture_allowed_when_quality_gates_red_but_echo_healthy(controller: CaptureController) -> None:
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
    session = controller.create_session("Eco saudável", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    saved = controller.capture()
    assert saved["saved"]


def _consistent_pair(index: int, x: float, y: float) -> tuple[VisionObservation, RobotPoseSnapshot]:
    center_x = 100.0 + 0.5 * x + 0.02 * y
    center_y = -30.0 - 0.01 * x + 0.45 * y
    visual_angle = (index * 23.0) % 180.0
    rz = (visual_angle + 14.0) % 180.0
    angle = math.radians(rz)
    robot_x = center_x + math.cos(angle) * 57.5
    robot_y = center_y + math.sin(angle) * 57.5
    vision = valid_vision(x, y)
    object.__setattr__(vision, "angle_deg", visual_angle)
    robot = valid_robot(robot_x, robot_y, 0.0, rz)
    return vision, robot


def test_rmse_feedback_after_three_adjustment_points(controller: CaptureController) -> None:
    session = controller.create_session("RMSE", [0.0])
    controller.activate_plan(session["id"], 0.0)
    samples = [(100.0, 100.0), (700.0, 100.0), (100.0, 400.0)]
    last = None
    for index, (x, y) in enumerate(samples):
        vision, robot = _consistent_pair(index, x, y)
        controller.vision_supplier = lambda vision=vision: vision
        controller.robot_supplier = lambda robot=robot: robot
        last = register_point(controller)
    assert last is not None
    assert last["feedback"]["rmse_available"] is True
    assert last["feedback"]["suggestion_kind"] == "ok"
    assert last["feedback"]["xy_rms_mm"] == pytest.approx(0.0, abs=1e-6)
    assert "Adequado" in last["feedback"]["suggestion"]


def test_next_available_point_index_fills_holes() -> None:
    from sincro_robo.services.capture import next_available_point_index

    assert next_available_point_index([]) == 1
    assert next_available_point_index([{"point_index": 1}, {"point_index": 2}]) == 3
    assert next_available_point_index(
        [{"point_index": n} for n in (1, 2, 3, 4, 5, 8, 9)]
    ) == 6
