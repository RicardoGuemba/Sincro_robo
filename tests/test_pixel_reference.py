from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sincro_robo.adapters.segmenter import annotate_frame
from sincro_robo.config import load_config
from sincro_robo.domain import RobotPoseSnapshot, VisionObservation, utc_now
from sincro_robo.geometry import (
    EstimatedMaskPose,
    MoldPoseEstimator,
    build_observation,
    configured_pixel_scales,
    map_to_reference,
    reference_scale_factors,
)
from sincro_robo.services.capture import CaptureController
from sincro_robo.storage import Storage

PICK_REFERENCE = {
    "source_width": 2592,
    "source_height": 1944,
    "destination_width": 960,
    "destination_height": 720,
}

QUALITY = {
    "min_confidence": 0.3,
    "min_axis_quality": 1.35,
    "min_mask_area_ratio": 0.002,
    "max_mask_area_ratio": 0.85,
}


def test_configured_scale_is_ten_over_twenty_seven() -> None:
    scale_x, scale_y = configured_pixel_scales(PICK_REFERENCE)
    assert scale_x == pytest.approx(10 / 27)
    assert scale_y == pytest.approx(10 / 27)
    assert scale_x == pytest.approx(960 / 2592)
    assert scale_y == pytest.approx(720 / 1944)


def test_acceptance_native_to_pick_coordinates() -> None:
    scale_x, scale_y = reference_scale_factors(2592, 1944, PICK_REFERENCE)
    x, y = map_to_reference(1281.5, 1028.9, scale_x, scale_y)
    assert x == pytest.approx(474.6296296296, abs=1e-6)
    assert y == pytest.approx(381.0740740741, abs=1e-6)
    assert round(x) == 475
    assert round(y) == 381


def test_already_960x720_is_identity() -> None:
    scale_x, scale_y = reference_scale_factors(960, 720, PICK_REFERENCE)
    assert (scale_x, scale_y) == (1.0, 1.0)
    x, y = map_to_reference(474.63, 381.07, scale_x, scale_y)
    assert x == pytest.approx(474.63)
    assert y == pytest.approx(381.07)


def test_unknown_frame_size_is_not_scaled() -> None:
    scale_x, scale_y = reference_scale_factors(960, 540, PICK_REFERENCE)
    assert (scale_x, scale_y) == (1.0, 1.0)


def test_mapping_is_applied_once_after_centroid() -> None:
    pose = EstimatedMaskPose(
        x=1281.5,
        y=1028.9,
        angle_deg=37.0,
        axis_vx=1.0,
        axis_vy=0.0,
        axis_quality=4.0,
        mask_area_ratio=0.1,
        mask_cut=False,
    )
    observation = build_observation(pose, 0.99, 1, (1944, 2592, 3), QUALITY, PICK_REFERENCE)
    assert observation.native_x == pytest.approx(1281.5)
    assert observation.native_y == pytest.approx(1028.9)
    assert observation.x == pytest.approx(1281.5 * 10 / 27)
    assert observation.y == pytest.approx(1028.9 * 10 / 27)
    assert observation.scale_x == pytest.approx(10 / 27)
    assert observation.scale_y == pytest.approx(10 / 27)
    assert observation.reference_width == 960
    assert observation.reference_height == 720
    assert observation.frame_width == 2592
    assert observation.frame_height == 1944
    assert observation.angle_deg == pytest.approx(0.0)
    remapped = map_to_reference(observation.x, observation.y, observation.scale_x, observation.scale_y)
    assert remapped[0] != pytest.approx(observation.x)
    overlay_x, overlay_y = observation.overlay_xy()
    assert overlay_x == pytest.approx(1281.5)
    assert overlay_y == pytest.approx(1028.9)


def test_estimator_centroid_stays_native() -> None:
    mask = np.zeros((1944, 2592), dtype=np.uint8)
    mask[1000:1060, 1250:1314] = 1
    pose = MoldPoseEstimator().estimate(mask)
    assert pose.x == pytest.approx(1281.5)
    assert pose.y == pytest.approx(1029.5)


def test_overlay_draws_native_point_not_pick_point() -> None:
    frame = np.zeros((1944, 2592, 3), dtype=np.uint8)
    mask = np.zeros((1944, 2592), dtype=bool)
    mask[1020:1038, 1272:1292] = True
    observation = build_observation(
        EstimatedMaskPose(1281.5, 1028.9, 0.0, 1.0, 0.0, 5.0, 0.1, False),
        0.99,
        1,
        frame.shape,
        QUALITY,
        PICK_REFERENCE,
    )
    overlay_x, overlay_y = observation.overlay_xy()
    annotated = annotate_frame(
        frame,
        mask,
        overlay_x,
        overlay_y,
        observation.angle_deg,
        major_axis_length=40.0,
    )
    native = tuple(int(round(value)) for value in (overlay_x, overlay_y))
    pick = tuple(int(round(value)) for value in (observation.x, observation.y))
    assert annotated[native[1], native[0]].sum() > 0
    assert annotated[pick[1], pick[0]].sum() == 0


def test_session_records_source_destination_and_scales(tmp_path: Path) -> None:
    config = load_config("config/pcbox.json")
    config["model"]["checkpoint"] = str(tmp_path / "missing.pth")
    config["app"]["database"] = str(tmp_path / "session.sqlite3")
    controller = CaptureController(
        Storage(config["app"]["database"]),
        config,
        lambda: None,
        lambda: None,
        tmp_path,
    )
    session = controller.create_session("Referência pick", [0.0, 200.0])
    reference = session["config"]["pixel_reference"]
    assert reference["source_width"] == 2592
    assert reference["source_height"] == 1944
    assert reference["destination_width"] == 960
    assert reference["destination_height"] == 720
    assert reference["scale_x"] == pytest.approx(10 / 27)
    assert reference["scale_y"] == pytest.approx(10 / 27)


def test_region_gate_uses_native_image_coordinates(tmp_path: Path) -> None:
    config = load_config("config/pcbox.json")
    config["model"]["checkpoint"] = str(tmp_path / "missing.pth")
    vision = VisionObservation(
        timestamp=utc_now(),
        x=1281.5 * 10 / 27,
        y=1028.9 * 10 / 27,
        angle_deg=25.0,
        confidence=0.99,
        axis_quality=4.2,
        mask_area_ratio=0.12,
        mask_cut=False,
        instance_count=1,
        stable=True,
        sigma_x=0.2,
        sigma_y=0.2,
        sigma_angle_deg=0.1,
        frame_width=2592,
        frame_height=1944,
        gates={
            "single_instance": True,
            "confidence": True,
            "mask_not_cut": True,
            "axis_quality": True,
            "mask_area": True,
        },
        native_x=1281.5,
        native_y=1028.9,
        scale_x=10 / 27,
        scale_y=10 / 27,
        reference_width=960,
        reference_height=720,
    )
    robot = RobotPoseSnapshot(utc_now(), 320.0, 210.0, 0.0, 180.0, 0.0, 37.0, True)
    controller = CaptureController(
        Storage(str(tmp_path / "region.sqlite3")),
        config,
        lambda: vision,
        lambda: robot,
        tmp_path,
    )
    session = controller.create_session("Região nativa", [0.0])
    controller.activate_plan(session["id"], 0.0)
    assert controller.capture_readiness()["region"]


def test_persisted_and_exported_coordinates_are_pick_space(tmp_path: Path) -> None:
    config = load_config("config/simulator.json")
    config["model"]["checkpoint"] = str(tmp_path / "missing.pth")
    config["app"]["exports_dir"] = str(tmp_path / "exports")
    pick_x = 1281.5 * 10 / 27
    pick_y = 1028.9 * 10 / 27
    vision = VisionObservation(
        timestamp=utc_now(),
        x=pick_x,
        y=pick_y,
        angle_deg=25.0,
        confidence=0.99,
        axis_quality=4.2,
        mask_area_ratio=0.12,
        mask_cut=False,
        instance_count=1,
        stable=True,
        sigma_x=0.2,
        sigma_y=0.2,
        sigma_angle_deg=0.1,
        frame_width=2592,
        frame_height=1944,
        gates={
            "single_instance": True,
            "confidence": True,
            "mask_not_cut": True,
            "axis_quality": True,
            "mask_area": True,
        },
        native_x=1281.5,
        native_y=1028.9,
        scale_x=10 / 27,
        scale_y=10 / 27,
        reference_width=960,
        reference_height=720,
    )
    robot = RobotPoseSnapshot(utc_now(), 320.0, 210.0, 0.0, 180.0, 0.0, 37.0, True)
    controller = CaptureController(
        Storage(str(tmp_path / "export.sqlite3")),
        config,
        lambda: vision,
        lambda: robot,
        tmp_path,
    )
    session = controller.create_session("Export pick", [0.0])
    controller.activate_plan(session["id"], 0.0)
    controller.capture()
    saved = controller.capture()
    pair = saved["pair"]
    assert pair["vision"]["x"] == pytest.approx(pick_x)
    assert pair["vision"]["y"] == pytest.approx(pick_y)
    assert pair["vision"]["native_x"] == pytest.approx(1281.5)
    exported = controller.export_session(session["id"], config["app"]["exports_dir"])
    csv_text = (tmp_path / "exports" / exported["csv"]).read_text(encoding="utf-8")
    assert f"{pick_x}"[:7] in csv_text or f"{pick_x:.1f}" in csv_text
    json_text = (tmp_path / "exports" / exported["json"]).read_text(encoding="utf-8")
    assert "native_x" in json_text
    assert "1281.5" in json_text
