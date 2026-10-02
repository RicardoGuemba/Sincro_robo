from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sincro_robo.api import create_app
from sincro_robo.config import load_config
from sincro_robo.domain import RobotPoseSnapshot, VisionObservation, utc_now
from sincro_robo.optics import (
    PATTERN,
    board_frame,
    detect_chessboard,
    direction_errors,
    object_points,
    render_checkerboard_frame,
    robot_from_camera,
    validation_metrics,
)
from sincro_robo.services.campaign import CampaignController
from sincro_robo.services.capture import CaptureController
from sincro_robo.storage import Storage


def _vision(x: float, y: float) -> VisionObservation:
    return VisionObservation(
        timestamp=utc_now(), x=x, y=y, angle_deg=25.0, confidence=0.99,
        axis_quality=4.2, mask_area_ratio=0.12, mask_cut=False,
        instance_count=1, stable=True, sigma_x=0.2, sigma_y=0.2,
        sigma_angle_deg=0.1, frame_width=960, frame_height=540,
        gates={
            "single_instance": True, "confidence": True, "mask_not_cut": True,
            "axis_quality": True, "mask_area": True, "vcp_scale": True,
        },
    )


def _robot() -> RobotPoseSnapshot:
    return RobotPoseSnapshot(utc_now(), 320.0, 210.0, 0.0, 180.0, 0.0, 37.0, True)


def _projected_views(count: int = 20) -> list[np.ndarray]:
    matrix = np.array([[900.0, 0.0, 480.0], [0.0, 900.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros((5, 1), dtype=np.float64)
    obj = object_points(PATTERN, 30.0)
    views: list[np.ndarray] = []
    for index in range(count):
        column = index % 3
        row = (index // 3) % 3
        center_x = (column + 0.5) * 960.0 / 3.0
        center_y = (row + 0.5) * 540.0 / 3.0
        rvec = np.array([[0.12 * np.sin(index), 0.1 * np.cos(index * 0.7), 0.03 * index]], dtype=np.float64)
        # O centro do tabuleiro 9x6 de 30 mm fica em (120, 75) no plano do objeto.
        tvec = np.array([[center_x - 480.0 - 120.0, center_y - 270.0 - 75.0, 900.0]], dtype=np.float64)
        image, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        views.append(image.reshape(-1, 2))
    return views


@pytest.fixture
def campaign(tmp_path: Path) -> CampaignController:
    config = load_config("config/simulator.json")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    storage = Storage(str(tmp_path / "campaign.sqlite3"))
    holder = {"vision": _vision(480.0, 270.0)}
    capture = CaptureController(
        storage, config, lambda: holder["vision"], _robot, tmp_path,
    )
    controller = CampaignController(storage, config, capture, tmp_path)
    controller._holder = holder  # type: ignore[attr-defined]
    return controller


def _hardware_checks(*, focus: bool) -> dict[str, bool]:
    return {
        "camera_fixed": True,
        "lens_focus_locked": focus,
        "production_resolution": True,
        "single_stapi_client": True,
    }


def _stabilize_focus(campaign: CampaignController) -> np.ndarray:
    frame = render_checkerboard_frame(960, 540, 0)
    for _ in range(10):
        campaign.observe_frame(frame)
    return frame


def test_advance_is_blocked_until_the_hardware_gate(campaign: CampaignController) -> None:
    campaign.create("Curitiba", 30.0)
    with pytest.raises(ValueError, match="nitidez"):
        campaign.advance()
    with pytest.raises(ValueError, match="nitidez"):
        campaign.set_checks(_hardware_checks(focus=True))
    _stabilize_focus(campaign)
    campaign.set_checks(_hardware_checks(focus=True))
    view = campaign.advance()
    assert view is not None
    assert view["stage"] == "capture_intrinsic"
    assert view["gate"]["passed"] is False
    assert view["focus"]["locked_peak"] > 1.0
    campaign.observe_frame(render_checkerboard_frame(960, 540, 2))
    later = campaign.focus_view()
    assert later["status"] == "Foco travado."
    assert later["peak"] == view["focus"]["locked_peak"]


def test_focus_drop_restarts_the_campaign(campaign: CampaignController) -> None:
    campaign.create("Foco", 30.0)
    sharp = _stabilize_focus(campaign)
    campaign.set_checks(_hardware_checks(focus=True))
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    for _ in range(10):
        campaign.observe_frame(blurred)
    view = campaign.view()
    assert view is not None
    assert view["focus"]["drift"] is True
    with pytest.raises(ValueError, match="foco"):
        campaign.advance()
    restarted = campaign.restart()
    assert restarted["stage"] == "fix_hardware"
    assert restarted["focus"]["drift"] is False
    assert restarted["focus"]["locked_peak"] is None


def test_checkerboard_renderer_is_detectable() -> None:
    found = 0
    for pose in range(8):
        frame = render_checkerboard_frame(960, 540, pose)
        detection = detect_chessboard(frame)
        found += int(bool(detection["found"]))
    assert found >= 6


def test_direction_and_plane_metrics_roundtrip() -> None:
    frame = board_frame([100.0, 200.0, 0.0], [400.0, 200.0, 0.0], [100.0, 450.0, 0.0])
    predicted = direction_errors([0.0, 0.0, 0.0], frame, 30.0)["predicted_mm"]
    errors = direction_errors(predicted, frame, 30.0)
    assert errors["error_correct_mm"] < 1e-6
    assert errors["error_inverted_mm"] > errors["error_correct_mm"]

    matrix = [[900.0, 0.0, 480.0], [0.0, 900.0, 270.0], [0.0, 0.0, 1.0]]
    distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
    extrinsic = {
        "object_to_camera": {
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 800.0],
        }
    }
    composed = robot_from_camera(extrinsic["object_to_camera"], frame)
    obj = object_points(PATTERN, 30.0)
    rotation = np.asarray(frame["R"], dtype=np.float64)
    origin = np.asarray(frame["t"], dtype=np.float64)
    expected = np.asarray([origin + rotation @ point for point in obj], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(np.asarray(composed["R"], dtype=np.float64))
    projected, _ = cv2.projectPoints(
        expected,
        rvec,
        np.asarray(composed["t"], dtype=np.float64).reshape(3, 1),
        np.asarray(matrix, dtype=np.float64),
        np.zeros((5, 1)),
    )
    metrics = validation_metrics(
        projected.reshape(-1, 2).tolist(),
        frame,
        composed,
        matrix,
        distortion,
        30.0,
        0.0,
    )
    assert metrics["millimetres"]["max"] < 1e-3


def test_full_wizard_reaches_the_mold_affine(campaign: CampaignController) -> None:
    campaign.create("Campanha local", 30.0)
    _stabilize_focus(campaign)
    campaign.set_checks(_hardware_checks(focus=True))
    campaign.advance()
    for corners in _projected_views(20):
        campaign.record_capture(corners.tolist(), (960, 540))
    view = campaign.advance()
    assert view is not None
    assert view["stage"] == "review_reprojection"
    campaign.solve_intrinsic()
    campaign.confirm_reprojection()
    campaign.advance()
    campaign.confirm_undistort()
    campaign.advance()
    campaign.record_frame_point("origin", {"x": 100.0, "y": 200.0, "z": 0.0})
    campaign.record_frame_point("plus_x", {"x": 400.0, "y": 200.0, "z": 0.0})
    campaign.record_frame_point("plus_y", {"x": 100.0, "y": 450.0, "z": 0.0})
    campaign.advance()
    campaign.record_capture(_projected_views(1)[0].tolist(), (960, 540))
    campaign.solve_pose()
    campaign.advance()
    taught = board_frame([100.0, 200.0, 0.0], [400.0, 200.0, 0.0], [100.0, 450.0, 0.0])
    corner = direction_errors([0.0, 0.0, 0.0], taught, 30.0)["predicted_mm"]
    campaign.record_direction_touch({"x": corner[0], "y": corner[1], "z": corner[2]})
    with pytest.raises(ValueError, match="invertido"):
        campaign.confirm_direction("inverted")
    campaign.confirm_direction("object_to_camera")
    campaign.advance()
    sample = _projected_views(1)[0].tolist()
    for plane in (0.0, 200.0, 400.0):
        campaign.select_validation_plane(plane)
        campaign.record_capture(sample, (960, 540), plan_z=plane)
    campaign.advance()
    assert campaign.view()["stage"] == "affine_mold"
    assert campaign.view()["profile_sha256"]
    profile_path = Path(campaign.project_root) / "data" / "campaigns" / campaign.active_id / "profile.json"
    assert profile_path.is_file()
    campaign.open_affine_session()
    capture = campaign.capture
    session_id = capture.active_session_id
    assert session_id is not None
    capture.activate_plan(session_id, 0.0)
    holder = campaign._holder  # type: ignore[attr-defined]
    targets = ((480.0, 270.0), (211.0, 130.0), (749.0, 130.0))
    for x, y in targets:
        holder["vision"] = _vision(x, y)
        capture.capture()
        capture.decide(True)
    view = campaign.advance()
    assert view is not None
    assert view["stage"] == "analyze_residuals"
    assert view["diagnosis"]["findings"]
    detail = capture.session_detail(session_id)
    assert detail["plans"][0]["evaluation"]["model"]["input_space"] == "undistorted_px_K_rect"
    assert detail["plans"][0]["evaluation"]["model"]["profile_sha256"] == view["profile_sha256"]


def test_campaign_api_blocks_skip(tmp_path: Path) -> None:
    config = load_config("config/simulator.json")
    config["app"]["database"] = str(tmp_path / "api.sqlite3")
    config["app"]["exports_dir"] = str(tmp_path / "exports")
    config["model"]["checkpoint"] = str(tmp_path / "not-needed.pth")
    with TestClient(create_app(config)) as client:
        created = client.post("/api/campaigns", json={"name": "API", "square_size_mm": 30})
        assert created.status_code == 201
        blocked = client.post("/api/campaigns/active/advance")
        assert blocked.status_code == 422
        state = client.get("/api/state")
        assert state.json()["campaign"]["stage"] == "fix_hardware"
