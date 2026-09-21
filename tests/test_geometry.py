from __future__ import annotations

import math

import numpy as np
import pytest

from sincro_robo.geometry import (
    MoldPoseEstimator,
    axis_angle_from_vector,
    normalize_axis_angle,
    signed_axis_delta,
)


def _axis_aligned_rect_mask(h: int, w: int, x: int, y: int, rw: int, rh: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + rh, x : x + rw] = 1
    return mask


def _rotated_rect_mask(
    h: int, w: int, cx: float, cy: float, rw: float, rh: float, angle_deg: float
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    theta = math.radians(-angle_deg)
    x_local = dx * math.cos(theta) - dy * math.sin(theta)
    y_local = dx * math.sin(theta) + dy * math.cos(theta)
    return ((np.abs(x_local) <= rw / 2) & (np.abs(y_local) <= rh / 2)).astype(np.uint8)


def _circular_diff(got: float, expected: float) -> float:
    delta = abs(float(got) - float(expected))
    return min(delta, 180.0 - delta)


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


def test_min_area_rect_uses_mask_centroid_and_major_side() -> None:
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[40:80, 30:170] = 1
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert pose.x == pytest.approx(99.5)
    assert pose.y == pytest.approx(59.5)
    assert pose.angle_deg == pytest.approx(0.0, abs=2.0)
    assert pose.axis_quality == pytest.approx(140.0 / 40.0, abs=0.2)
    assert pose.major_axis_length == pytest.approx(140.0, abs=2.0)
    assert pose.minor_axis_length == pytest.approx(40.0, abs=2.0)
    assert not pose.mask_cut


def test_angle_90_for_vertical_rect() -> None:
    mask = _axis_aligned_rect_mask(200, 100, x=45, y=10, rw=10, rh=180)
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert abs(pose.angle_deg - 90.0) < 2.0
    assert pose.axis_quality > 3.0


def test_angle_comes_from_min_area_rect_not_axis_aligned_bbox() -> None:
    mask = _rotated_rect_mask(300, 300, cx=150, cy=150, rw=120, rh=25, angle_deg=45)
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    width = np.where(mask.any(axis=0))[0]
    height = np.where(mask.any(axis=1))[0]
    aabb_w = int(width[-1] - width[0] + 1)
    aabb_h = int(height[-1] - height[0] + 1)
    aabb_angle = 0.0 if aabb_w >= aabb_h else 90.0
    assert _circular_diff(pose.angle_deg, 45.0) < 5.0
    assert _circular_diff(pose.angle_deg, aabb_angle) > 20.0


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 75, 120, 150])
def test_angle_matches_rotated_rect(angle: float) -> None:
    mask = _rotated_rect_mask(300, 300, cx=150, cy=150, rw=120, rh=25, angle_deg=angle)
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert _circular_diff(pose.angle_deg, angle % 180.0) < 5.0


def test_pca_fallback_when_min_area_rect_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[40:80, 30:170] = 1
    monkeypatch.setattr("sincro_robo.geometry._angle_and_sides_from_min_area_rect", lambda _mask: None)
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert pose.x == pytest.approx(99.5)
    assert pose.y == pytest.approx(59.5)
    assert pose.angle_deg == pytest.approx(0.0, abs=2.0)
    assert pose.axis_quality > 1.35
    assert pose.major_axis_length > pose.minor_axis_length


def test_disc_centroid_is_center() -> None:
    yy, xx = np.mgrid[0:200, 0:200]
    mask = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 50 ** 2).astype(np.uint8)
    pose = MoldPoseEstimator(border_margin_px=2).estimate(mask)
    assert pose.x == pytest.approx(100.0, abs=1.0)
    assert pose.y == pytest.approx(100.0, abs=1.0)
    assert pose.axis_quality == pytest.approx(1.0, abs=0.15)


def test_mask_touching_border_is_rejected() -> None:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[0:20, 20:60] = 1
    assert MoldPoseEstimator(border_margin_px=3).estimate(mask).mask_cut


DEFAULT_ROI = [66, 64, 841, 615]


def _color_hits(patch: np.ndarray, color, tolerance: int = 8) -> bool:
    target = np.asarray(color, dtype=np.int16)
    if patch.ndim == 1:
        return bool(np.max(np.abs(patch.astype(np.int16) - target)) <= tolerance)
    return bool(np.any(np.max(np.abs(patch.astype(np.int16) - target), axis=-1) <= tolerance))


def _neighborhood_has_color(img: np.ndarray, y: int, x: int, color, radius: int = 3) -> bool:
    height, width = img.shape[:2]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    return _color_hits(img[y0:y1, x0:x1], color)


@pytest.mark.parametrize(
    ("heading", "cx", "cy", "expected"),
    [
        (0.0, 321.0, 166.0, (376.0, 166.0)),
        (90.0, 233.0, 266.0, (233.0, 211.0)),
        (91.0, 423.0, 256.0, (422.0, 201.0)),
        (30.0, 324.0, 227.0, (372.0, 200.0)),
        (180.0, 321.0, 166.0, (266.0, 166.0)),
    ],
)
def test_vcpn_gold_cases_keep_55mm_hypotenuse(
    heading: float, cx: float, cy: float, expected: tuple[float, float]
) -> None:
    from sincro_robo.geometry import vcpn_from_heading

    vcpn = vcpn_from_heading(cx, cy, heading, 55.0, 1.0)
    assert vcpn[0] == pytest.approx(expected[0], abs=1.0)
    assert vcpn[1] == pytest.approx(expected[1], abs=1.0)
    assert math.hypot(vcpn[0] - cx, vcpn[1] - cy) == pytest.approx(55.0, abs=1e-6)
    if heading in {0.0, 180.0}:
        assert vcpn[1] == pytest.approx(cy)
    else:
        assert vcpn[1] < cy


def test_vector_never_points_south() -> None:
    from sincro_robo.geometry import vcpn_from_heading

    for heading in (0.0, 15.0, 45.0, 90.0, 135.0, 180.0):
        vx, vy = vcpn_from_heading(400.0, 300.0, heading, 55.0, 1.0)
        assert vy <= 300.0 + 1e-9


def test_heading_keeps_180_after_west_flip() -> None:
    from sincro_robo.geometry import heading_north_deg, orient_north

    assert orient_north(-1.0, 0.0) == (-1.0, 0.0)
    assert heading_north_deg(-1.0, 0.0) == pytest.approx(180.0)
    assert heading_north_deg(1.0, 0.0) == pytest.approx(0.0)


def test_opposite_axis_vectors_share_vcpn() -> None:
    from sincro_robo.geometry import heading_north_deg, vcpn_from_heading

    heading_south = heading_north_deg(0.0, 1.0)
    heading_north = heading_north_deg(0.0, -1.0)
    assert heading_south == pytest.approx(heading_north)
    assert heading_south == pytest.approx(90.0)
    south = vcpn_from_heading(233.0, 266.0, heading_south, 55.0, 1.0)
    north = vcpn_from_heading(233.0, 266.0, heading_north, 55.0, 1.0)
    assert south == pytest.approx(north)
    assert south == pytest.approx((233.0, 211.0))


def test_zero_offset_returns_centroid() -> None:
    from sincro_robo.geometry import vcpn_from_heading

    assert vcpn_from_heading(321.0, 166.0, 91.0, 0.0, 1.0) == (321.0, 166.0)


def test_default_roi_compass_is_top_mid() -> None:
    from sincro_robo.geometry import DEFAULT_ROI_PX, roi_compass_anchor

    assert list(DEFAULT_ROI_PX) == [66.0, 64.0, 841.0, 615.0]
    assert roi_compass_anchor(DEFAULT_ROI_PX) == pytest.approx((486.5, 64.0))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (700.0, 150.0, "NE"),
        (200.0, 150.0, "NO"),
        (700.0, 500.0, "SE"),
        (200.0, 500.0, "SO"),
        (486.5, 371.5, "SE"),
        (50.0, 200.0, None),
        (920.0, 200.0, None),
        (400.0, 20.0, None),
        (400.0, 700.0, None),
    ],
)
def test_roi_quadrants_and_outside(x: float, y: float, expected: str | None) -> None:
    from sincro_robo.geometry import roi_quadrant_of_point

    assert roi_quadrant_of_point(x, y, DEFAULT_ROI) == expected


def test_quadrant_does_not_flip_cathetus_sign() -> None:
    from sincro_robo.geometry import roi_quadrant_of_point, vcpn_from_heading

    cx, cy = 400.0, 260.0
    vcpn = vcpn_from_heading(cx, cy, 90.0, 55.0, 1.0)
    assert vcpn[1] < cy
    assert roi_quadrant_of_point(*vcpn, DEFAULT_ROI) in {"NE", "NO", "SE", "SO", None}


def test_hud_ascii_matches_supervisory_example() -> None:
    from sincro_robo.overlay import format_vcpn_hud

    lines = format_vcpn_hud(
        423,
        256,
        91,
        422,
        201,
        confidence=0.99,
        roi_quadrant="NE",
        mask_area_cm2=152.0,
    )
    assert lines == [
        "Embalagem",
        "conf:99%",
        "C",
        "CX:423",
        "CY:256",
        "Vetor",
        "ang:91deg",
        "VCPn",
        "X:422",
        "Y:201",
        "Q:NE",
        "A:152.0cm2",
    ]


def test_vcpn_overlay_draws_roi_compass_centroid_arrow_and_hud() -> None:
    from sincro_robo.adapters.segmenter import annotate_frame
    from sincro_robo.geometry import vcpn_from_heading
    from sincro_robo.overlay import (
        ROI_COMPASS_COLOR_BGR,
        ROI_RECT_COLOR_BGR,
        VCPN_ARROW_COLOR_BGR,
        VCPN_CENTROID_COLOR_BGR,
        VCPN_POINT_COLOR_BGR,
        format_vcpn_hud,
    )

    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    mask = np.zeros((720, 960), dtype=bool)
    mask[230:280, 400:450] = True
    cx, cy, heading = 423.0, 256.0, 91.0
    vx, vy = vcpn_from_heading(cx, cy, heading, 55.0, 1.0)
    out = annotate_frame(
        frame,
        mask,
        cx,
        cy,
        heading,
        vcpn_x=vx,
        vcpn_y=vy,
        roi_px=DEFAULT_ROI,
        roi_enabled=True,
        roi_quadrant="NE",
        confidence=0.99,
        mask_area_cm2=152.0,
        overlay="vcpn",
    )
    assert _neighborhood_has_color(out, 64, 66, ROI_RECT_COLOR_BGR, radius=2)
    assert _neighborhood_has_color(out, 64, 907, ROI_RECT_COLOR_BGR, radius=2)
    assert _neighborhood_has_color(out, 64, 487, ROI_COMPASS_COLOR_BGR, radius=4)
    assert _neighborhood_has_color(out, int(round(cy)), int(round(cx)), VCPN_CENTROID_COLOR_BGR)
    assert _neighborhood_has_color(out, int(round(vy)), int(round(vx)), VCPN_POINT_COLOR_BGR)
    mid_y = int(round((cy + vy) / 2.0))
    mid_x = int(round((cx + vx) / 2.0))
    assert _neighborhood_has_color(out, mid_y, mid_x, VCPN_ARROW_COLOR_BGR, radius=6)
    hud = "\n".join(format_vcpn_hud(cx, cy, heading, vx, vy, confidence=0.99, roi_quadrant="NE", mask_area_cm2=152.0))
    assert "C" in hud and "Vetor" in hud and "VCPn" in hud and "Q:NE" in hud
    assert out[10:140, 8:120].sum() > 0


def test_config_default_roi_and_vcp_offset() -> None:
    from sincro_robo.config import DEFAULT_CONFIG

    vision = DEFAULT_CONFIG["vision_reference"]
    assert vision["vcp_offset_mm"] == 55.0
    assert vision["mm_per_px"] == 1.026
    assert vision["roi_enabled"] is True
    assert vision["roi_px"] == [66, 64, 841, 615]
    assert DEFAULT_CONFIG["calibration"]["pick_offset_local_mm"] == [0.0, 55.0]

