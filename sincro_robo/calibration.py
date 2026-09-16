from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians, sin, sqrt
from typing import Any, Iterable

import numpy as np

from .geometry import normalize_axis_angle, signed_axis_delta


def robot_geometric_center(
    robot_x: float,
    robot_y: float,
    robot_rz_deg: float,
    pick_offset_local_mm: Iterable[float] = (57.5, 0.0),
) -> tuple[float, float]:
    dx, dy = (float(value) for value in pick_offset_local_mm)
    angle = radians(float(robot_rz_deg))
    rotated_x = cos(angle) * dx - sin(angle) * dy
    rotated_y = sin(angle) * dx + cos(angle) * dy
    return float(robot_x) - rotated_x, float(robot_y) - rotated_y


def circular_axis_mean_deg(values: Iterable[float]) -> float:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("Não há ângulos para calcular a média")
    doubled = np.radians(values_array * 2.0)
    return normalize_axis_angle(
        np.degrees(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2.0
    )


@dataclass(frozen=True)
class CalibrationModel:
    plan_z: float
    affine: list[list[float]]
    angle_offset_deg: float
    adjustment_count: int

    def transform_xy(self, x: float, y: float) -> tuple[float, float]:
        matrix = np.asarray(self.affine, dtype=np.float64)
        result = np.asarray([float(x), float(y), 1.0]) @ matrix
        return float(result[0]), float(result[1])

    def transform_angle(self, visual_angle_deg: float) -> float:
        return normalize_axis_angle(float(visual_angle_deg) + self.angle_offset_deg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_plan_calibration(
    plan_z: float,
    pairs: Iterable[dict[str, Any]],
    pick_offset_local_mm: Iterable[float] = (57.5, 0.0),
) -> CalibrationModel:
    adjustment = [pair for pair in pairs if pair.get("role") == "adjustment"]
    if len(adjustment) < 3:
        raise ValueError("São necessários pelo menos 3 pontos de ajuste")

    source: list[list[float]] = []
    target: list[list[float]] = []
    angle_offsets: list[float] = []
    for pair in adjustment:
        vision = pair["vision"]
        robot = pair["robot"]
        center_x, center_y = robot_geometric_center(
            robot["x"], robot["y"], robot["rz"], pick_offset_local_mm
        )
        source.append([float(vision["x"]), float(vision["y"]), 1.0])
        target.append([center_x, center_y])
        angle_offsets.append(float(robot["rz"]) - float(vision["angle_deg"]))

    matrix, _residuals, rank, _singular = np.linalg.lstsq(
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        rcond=None,
    )
    if rank < 3:
        raise ValueError("Pontos de ajuste degenerados ou alinhados")
    return CalibrationModel(
        plan_z=float(plan_z),
        affine=matrix.tolist(),
        angle_offset_deg=circular_axis_mean_deg(angle_offsets),
        adjustment_count=len(adjustment),
    )


def _stats(values: list[float], include_p95: bool = True) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(array.mean()),
        "rms": float(sqrt(float(np.mean(np.square(array))))),
        "max": float(array.max()),
    }
    if include_p95:
        result["p95"] = float(np.percentile(array, 95))
    return result


def evaluate_plan(
    model: CalibrationModel,
    pairs: Iterable[dict[str, Any]],
    pick_offset_local_mm: Iterable[float] = (57.5, 0.0),
    xy_tolerance_mm: float = 20.0,
    angular_tolerance_deg: float = 5.0,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    validation_xy: list[float] = []
    validation_angle: list[float] = []
    all_xy: list[float] = []
    all_angle: list[float] = []
    for pair in pairs:
        vision = pair["vision"]
        robot = pair["robot"]
        expected_x, expected_y = robot_geometric_center(
            robot["x"], robot["y"], robot["rz"], pick_offset_local_mm
        )
        predicted_x, predicted_y = model.transform_xy(vision["x"], vision["y"])
        error_xy = float(np.hypot(predicted_x - expected_x, predicted_y - expected_y))
        predicted_angle = model.transform_angle(vision["angle_deg"])
        error_angle = abs(signed_axis_delta(predicted_angle, robot["rz"]))
        role = pair.get("role", "adjustment")
        all_xy.append(error_xy)
        all_angle.append(error_angle)
        if role == "validation":
            validation_xy.append(error_xy)
            validation_angle.append(error_angle)
        evaluated.append(
            {
                "pair_id": pair.get("id"),
                "role": role,
                "residual_xy_mm": error_xy,
                "error_angle_deg": error_angle,
                "status": "suspect"
                if error_xy > xy_tolerance_mm or error_angle > angular_tolerance_deg
                else "accepted",
            }
        )

    metric_xy = validation_xy or all_xy
    metric_angle = validation_angle or all_angle
    xy = _stats(metric_xy, include_p95=True)
    angle = _stats(metric_angle, include_p95=False)
    passed = bool(metric_xy) and xy["max"] <= xy_tolerance_mm and angle["max"] <= angular_tolerance_deg
    return {
        "xy_mm": xy,
        "angle_deg": angle,
        "validation_count": len(validation_xy),
        "evaluated_count": len(evaluated),
        "passed": passed,
        "limits": {"xy_mm": float(xy_tolerance_mm), "angle_deg": float(angular_tolerance_deg)},
        "pairs": evaluated,
    }

