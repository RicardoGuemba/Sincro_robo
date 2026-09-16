from __future__ import annotations

import math

import pytest

from sincro_robo.calibration import (
    evaluate_plan,
    fit_plan_calibration,
    robot_geometric_center,
)


def _pair(index: int, x: float, y: float, role: str = "adjustment") -> dict:
    center_x = 100.0 + 0.5 * x + 0.02 * y
    center_y = -30.0 - 0.01 * x + 0.45 * y
    visual_angle = (index * 23.0) % 180.0
    rz = (visual_angle + 14.0) % 180.0
    angle = math.radians(rz)
    robot_x = center_x + math.cos(angle) * 57.5
    robot_y = center_y + math.sin(angle) * 57.5
    return {
        "id": f"p{index}",
        "role": role,
        "vision": {"x": x, "y": y, "angle_deg": visual_angle},
        "robot": {"x": robot_x, "y": robot_y, "rz": rz},
    }


def test_pick_offset_rotates_with_robot_rz() -> None:
    assert robot_geometric_center(157.5, 200.0, 0.0) == pytest.approx((100.0, 200.0))
    assert robot_geometric_center(100.0, 257.5, 90.0) == pytest.approx((100.0, 200.0))


def test_affine_fit_and_validation_metrics() -> None:
    pairs = [
        _pair(0, 100, 100),
        _pair(1, 700, 100),
        _pair(2, 100, 400),
        _pair(3, 700, 400),
        _pair(4, 400, 250),
        _pair(5, 250, 210, "validation"),
        _pair(6, 590, 330, "validation"),
    ]
    model = fit_plan_calibration(200.0, pairs)
    assert model.transform_xy(321.0, 222.0) == pytest.approx(
        (100.0 + 0.5 * 321.0 + 0.02 * 222.0, -30.0 - 0.01 * 321.0 + 0.45 * 222.0)
    )
    assert model.angle_offset_deg == pytest.approx(14.0)
    metrics = evaluate_plan(model, pairs)
    assert metrics["passed"]
    assert metrics["validation_count"] == 2
    assert metrics["xy_mm"]["max"] < 1e-8
    assert metrics["angle_deg"]["max"] < 1e-8


def test_degenerate_points_are_refused() -> None:
    pairs = [_pair(index, float(index), float(index)) for index in range(3)]
    with pytest.raises(ValueError, match="degenerados"):
        fit_plan_calibration(0.0, pairs)

