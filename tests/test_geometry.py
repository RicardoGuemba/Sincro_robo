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
    assert pose.angle_deg == pytest.approx(0.0)
    assert pose.axis_quality > 10
    assert not pose.mask_cut


def test_mask_touching_border_is_rejected() -> None:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[0:20, 20:60] = 1
    assert MoldPoseEstimator(border_margin_px=3).estimate(mask).mask_cut

