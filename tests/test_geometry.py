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
