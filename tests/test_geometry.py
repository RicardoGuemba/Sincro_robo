from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from sincro_robo.adapters.segmenter import annotate_frame
from sincro_robo.config import DEFAULT_CONFIG
from sincro_robo.geometry import (
    EstimatedMaskPose,
    MoldPoseEstimator,
    axis_angle_from_vector,
    build_observation,
    heading_north_deg,
    normalize_axis_angle,
    orient_north,
    roi_quadrant_of_point,
    resolve_vcp_scale_px,
    signed_axis_delta,
    vcpn_from_heading,
)

QUALITY = {
    "min_confidence": 0.3,
    "min_axis_quality": 1.35,
    "min_mask_area_ratio": 0.002,
    "max_mask_area_ratio": 0.85,
}


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (90.0, 90.0), (179.0, 179.0), (180.0, 0.0), (359.0, 179.0)],
)
def test_normalize_required_angles(angle: float, expected: float) -> None:
    assert normalize_axis_angle(angle) == pytest.approx(expected)


def test_axis_equivalence_after_vector_sign_change() -> None:
    for degrees_value in (0.0, 37.0, 90.0, 179.0):
        radians_value = math.radians(degrees_value)
        vector = (math.cos(radians_value), math.sin(radians_value))
        forward = axis_angle_from_vector(*vector)
        backward = axis_angle_from_vector(-vector[0], -vector[1])
        assert forward == pytest.approx(backward)


def test_signed_axis_delta_wraps_at_180() -> None:
    assert signed_axis_delta(1.0, 179.0) == pytest.approx(2.0)
    assert signed_axis_delta(179.0, 1.0) == pytest.approx(-2.0)


def test_pca_uses_mask_centroid_and_major_axis() -> None:
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[40:80, 30:170] = 1
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert pose.x == pytest.approx(99.5)
    assert pose.y == pytest.approx(59.5)
    assert pose.angle_deg == pytest.approx(0.0) or pose.angle_deg == pytest.approx(180.0)
    assert pose.axis_vy == pytest.approx(0.0, abs=1e-9)
    assert pose.axis_quality > 10
    assert not pose.mask_cut


def test_mask_touching_border_is_rejected() -> None:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[0:20, 20:60] = 1
    assert MoldPoseEstimator(border_margin_px=3).estimate(mask).mask_cut


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ((1.0, 0.0), (1.0, 0.0)),
        ((0.0, 1.0), (0.0, -1.0)),
        ((0.0, -1.0), (0.0, -1.0)),
        ((-1.0, 0.0), (-1.0, 0.0)),
        ((0.7, 0.7), (-math.sqrt(0.5), -math.sqrt(0.5))),
    ],
)
def test_orient_north_never_points_south(
    vector: tuple[float, float], expected: tuple[float, float]
) -> None:
    ux, uy = orient_north(*vector)
    assert uy <= 1e-12
    assert math.hypot(ux, uy) == pytest.approx(1.0)
    assert ux == pytest.approx(expected[0], abs=1e-9)
    assert uy == pytest.approx(expected[1], abs=1e-9)


def test_south_and_north_pca_share_heading_and_vcpn() -> None:
    cx, cy, s_px = 400.0, 300.0, 55.0
    south = heading_north_deg(0.0, 1.0)
    north = heading_north_deg(0.0, -1.0)
    assert south == pytest.approx(90.0)
    assert north == pytest.approx(90.0)
    assert vcpn_from_heading(cx, cy, south, s_px) == pytest.approx((cx, cy - s_px))
    assert vcpn_from_heading(cx, cy, north, s_px) == pytest.approx((cx, cy - s_px))


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        (0.0, (455.0, 300.0)),
        (90.0, (400.0, 245.0)),
        (180.0, (345.0, 300.0)),
    ],
)
def test_vcpn_cardinals(heading: float, expected: tuple[float, float]) -> None:
    assert vcpn_from_heading(400.0, 300.0, heading, 55.0) == pytest.approx(expected)


@pytest.mark.parametrize("heading", [60.0, 120.0])
def test_vcpn_catheti_match_hypotenuse(heading: float) -> None:
    cx, cy, s_px = 400.0, 300.0, 55.0
    vcpn_x, vcpn_y = vcpn_from_heading(cx, cy, heading, s_px)
    delta_x = s_px * math.cos(math.radians(heading))
    delta_y = -s_px * math.sin(math.radians(heading))
    assert vcpn_x - cx == pytest.approx(delta_x)
    assert vcpn_y - cy == pytest.approx(delta_y)
    assert math.hypot(vcpn_x - cx, vcpn_y - cy) == pytest.approx(s_px)


@pytest.mark.parametrize("heading", [0.0, 15.0, 45.0, 90.0, 135.0, 165.0, 180.0])
def test_vcpn_stays_on_or_north_of_centroid(heading: float) -> None:
    vcpn_x, vcpn_y = vcpn_from_heading(100.0, 80.0, heading, 55.0)
    assert vcpn_y <= 80.0 + 1e-9
    assert math.hypot(vcpn_x - 100.0, vcpn_y - 80.0) == pytest.approx(55.0)


def test_zero_offset_returns_centroid() -> None:
    assert vcpn_from_heading(120.0, 80.0, 60.0, 0.0) == pytest.approx((120.0, 80.0))


def test_heading_cardinals() -> None:
    assert heading_north_deg(1.0, 0.0) == pytest.approx(0.0)
    assert heading_north_deg(0.0, -1.0) == pytest.approx(90.0)
    assert heading_north_deg(-1.0, 0.0) == pytest.approx(180.0)
    assert 0.0 <= heading_north_deg(0.5, -0.5) <= 180.0


def test_vertical_mask_heading_is_north() -> None:
    mask = np.zeros((200, 120), dtype=np.uint8)
    mask[20:180, 45:75] = 1
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert pose.angle_deg == pytest.approx(90.0)
    assert pose.axis_vy < 0.0
    assert pose.axis_vx == pytest.approx(0.0, abs=1e-6)


def test_build_observation_computes_vcpn_from_catheti() -> None:
    pose = EstimatedMaskPose(
        x=400.0, y=300.0, angle_deg=60.0, axis_vx=0.5, axis_vy=-math.sqrt(3) / 2,
        axis_quality=4.0, mask_area_ratio=0.1, mask_cut=False,
    )
    observation = build_observation(
        pose, 0.9, 1, (540, 960), QUALITY, {"vcp_offset_mm": 55.0, "mm_per_px": 1.0}
    )
    expected_x = 400.0 + 55.0 * math.cos(math.radians(60.0))
    expected_y = 300.0 - 55.0 * math.sin(math.radians(60.0))
    assert observation.x == pytest.approx(400.0)
    assert observation.y == pytest.approx(300.0)
    assert observation.vcpn_x == pytest.approx(expected_x)
    assert observation.vcpn_y == pytest.approx(expected_y)
    assert observation.vcpn_y < observation.y
    assert observation.gates["vcp_scale"]


@pytest.mark.parametrize("mm_per_px", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_fails_vcp_gate(mm_per_px: float) -> None:
    pose = EstimatedMaskPose(
        x=10.0, y=20.0, angle_deg=90.0, axis_vx=0.0, axis_vy=-1.0,
        axis_quality=4.0, mask_area_ratio=0.1, mask_cut=False,
    )
    observation = build_observation(
        pose, 0.9, 1, (540, 960), QUALITY, {"vcp_offset_mm": 55.0, "mm_per_px": mm_per_px}
    )
    assert observation.gates["vcp_scale"] is False
    assert observation.vcpn_x == pytest.approx(pose.x)
    assert observation.vcpn_y == pytest.approx(pose.y)


def test_resolve_vcp_scale_rejects_non_positive() -> None:
    assert resolve_vcp_scale_px(55.0, 1.0) == pytest.approx(55.0)
    assert resolve_vcp_scale_px(55.0, 0.0) is None
    assert resolve_vcp_scale_px(0.0, 1.0) == pytest.approx(0.0)


def test_vcpn_defaults_match_the_poc_usb_roi_and_operational_offset() -> None:
    reference = DEFAULT_CONFIG["vision_reference"]
    assert reference["vcp_offset_mm"] == pytest.approx(55.0)
    assert reference["roi_enabled"] is True
    assert reference["roi_px"] == [189, 103, 277, 277]
    assert DEFAULT_CONFIG["calibration"]["pick_offset_local_mm"] == [55.0, 0.0]


@pytest.mark.parametrize(
    ("heading", "center", "expected_rounded"),
    [
        (0.0, (321.0, 166.0), (376, 166)),
        (90.0, (233.0, 266.0), (233, 211)),
        (180.0, (321.0, 166.0), (266, 166)),
        (30.0, (324.0, 227.0), (372, 200)),
        (91.0, (423.0, 256.0), (422, 201)),
    ],
)
def test_vcpn_golden_cases_keep_a_55_mm_hypotenuse(
    heading: float, center: tuple[float, float], expected_rounded: tuple[int, int]
) -> None:
    vcpn_x, vcpn_y = vcpn_from_heading(*center, heading, 55.0)
    assert (round(vcpn_x), round(vcpn_y)) == expected_rounded
    assert math.hypot(vcpn_x - center[0], vcpn_y - center[1]) == pytest.approx(55.0)
    if heading in (0.0, 180.0):
        assert vcpn_y == pytest.approx(center[1])
    else:
        assert vcpn_y < center[1]


def test_opposite_pca_axis_signs_produce_the_same_north_vcpn() -> None:
    north_axis = (math.cos(math.radians(30.0)), -math.sin(math.radians(30.0)))
    south_axis = (-north_axis[0], -north_axis[1])
    north_heading = heading_north_deg(*north_axis)
    south_heading = heading_north_deg(*south_axis)
    assert north_heading == pytest.approx(30.0)
    assert south_heading == pytest.approx(30.0)
    assert vcpn_from_heading(324.0, 227.0, north_heading, 55.0) == pytest.approx(
        vcpn_from_heading(324.0, 227.0, south_heading, 55.0)
    )


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((327.5, 103.0), "NE"),
        ((327.49, 103.0), "NO"),
        ((327.5, 241.5), "SE"),
        ((327.49, 241.5), "SO"),
        ((188.99, 241.5), None),
        ((466.01, 241.5), None),
        ((327.5, 102.99), None),
        ((327.5, 380.01), None),
    ],
)
def test_roi_quadrant_classifies_vcpn_without_clamping(
    point: tuple[float, float], expected: str | None
) -> None:
    assert roi_quadrant_of_point(*point, [189, 103, 277, 277]) == expected


def test_build_observation_publishes_roi_quadrant_only_when_enabled() -> None:
    pose = EstimatedMaskPose(
        x=423.0, y=256.0, angle_deg=91.0, axis_vx=-math.sin(math.radians(1.0)),
        axis_vy=-math.cos(math.radians(1.0)), axis_quality=4.0,
        mask_area_ratio=0.1, mask_cut=False,
    )
    enabled = build_observation(
        pose, 0.99, 1, (540, 960), QUALITY,
        {"vcp_offset_mm": 55.0, "mm_per_px": 1.0, "roi_enabled": True, "roi_px": [189, 103, 277, 277]},
    )
    disabled = build_observation(
        pose, 0.99, 1, (540, 960), QUALITY,
        {"vcp_offset_mm": 55.0, "mm_per_px": 1.0, "roi_enabled": False, "roi_px": [189, 103, 277, 277]},
    )
    assert enabled.roi_quadrant == "NE"
    assert disabled.roi_quadrant is None


def test_overlay_draws_roi_vcpn_markers_and_ascii_hud(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[220:292, 395:451] = 1
    labels: list[str] = []
    original_put_text = cv2.putText

    def record_put_text(image: np.ndarray, text: str, *args: object, **kwargs: object) -> np.ndarray:
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_put_text)
    annotated = annotate_frame(
        frame,
        mask,
        x=423.0,
        y=256.0,
        angle_deg=91.0,
        vcpn_x=422.0,
        vcpn_y=201.0,
        roi_enabled=True,
        roi_px=[189, 103, 277, 277],
        label="Embalagem",
        confidence=0.99,
        mask_area_cm2=152.0,
        roi_quadrant="NE",
    )

    assert annotated.shape == frame.shape
    assert annotated[103, 189, 1] > 200  # ROI verde
    assert tuple(annotated[103, 328]) == (255, 0, 255)  # rosa N/L em (327.5, 103)
    assert tuple(annotated[256, 423]) == (255, 255, 255)  # C branco
    assert tuple(annotated[201, 422]) == (0, 0, 255)  # VCPn vermelho
    assert annotated[230, 422, 1] > 200 and annotated[230, 422, 2] > 200  # seta amarela
    for line in ("N", "L", "Embalagem", "conf:99%", "C", "CX:423", "CY:256", "Vetor", "ang:91deg", "VCPn", "X:422", "Y:201", "Q:NE", "A:152.0cm2"):
        assert line in labels
