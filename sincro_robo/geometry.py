from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt
from typing import Any, Iterable, Sequence

import numpy as np

from .domain import VisionObservation, utc_now

_HORIZONTAL_EPS = 1e-12


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected image axis to the continuous range [0, 180)."""
    normalized = float(angle_deg) % 180.0
    return 0.0 if abs(normalized - 180.0) < 1e-10 else normalized


def signed_axis_delta(angle_deg: float, reference_deg: float) -> float:
    """Smallest signed delta for axes where theta and theta + 180 are equal."""
    return ((float(angle_deg) - float(reference_deg) + 90.0) % 180.0) - 90.0


def directed_heading_delta(angle_deg: float, reference_deg: float) -> float:
    """Signed delta for a north-directed heading in [0, 180]. East and west differ."""
    return float(angle_deg) - float(reference_deg)


def axis_angle_from_vector(vx: float, vy: float) -> float:
    # Image coordinates grow downwards, so atan2(vy, vx) is clockwise-positive.
    return normalize_axis_angle(degrees(atan2(float(vy), float(vx))))


def orient_north(ux: float, uy: float) -> tuple[float, float]:
    """Return a unit vector in the northern image half-plane (uy <= 0)."""
    vx, vy = float(ux), float(uy)
    norm = hypot(vx, vy)
    if norm <= _HORIZONTAL_EPS:
        return 0.0, -1.0
    vx /= norm
    vy /= norm
    if vy > 0.0:
        vx, vy = -vx, -vy
    return vx, vy


def heading_north_deg(ux: float, uy: float) -> float:
    """Heading in [0, 180]: 0° east, 90° north, 180° west."""
    vx, vy = orient_north(ux, uy)
    angle = degrees(atan2(-vy, vx))
    if angle < 0.0 and angle > -1e-10:
        return 0.0
    if angle < 0.0:
        angle += 360.0
    if angle > 180.0 and angle < 180.0 + 1e-10:
        return 180.0
    return min(180.0, max(0.0, angle))


def vcpn_from_heading(
    cx: float, cy: float, heading_deg: float, s_px: float
) -> tuple[float, float]:
    """VCPn from the right-triangle legs of a  hypotenuse along the north heading."""
    theta = radians(float(heading_deg))
    offset = float(s_px)
    return float(cx) + offset * cos(theta), float(cy) - offset * sin(theta)


def roi_quadrant_of_point(
    vcpn_x: float,
    vcpn_y: float,
    roi_px: Sequence[float] | None,
) -> str | None:
    """Classify a VCPn point in the configured image ROI, without moving it.

    The ROI uses image-space ``xywh`` coordinates: X grows east/right and Y grows
    south/down. Its border is considered part of the rectangle. A point outside
    the rectangle has no quadrant rather than being clamped to one.
    """
    if roi_px is None:
        return None
    try:
        roi_values = tuple(roi_px)
    except TypeError:
        return None
    if len(roi_values) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in roi_values)
        point_x = float(vcpn_x)
        point_y = float(vcpn_y)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in (x, y, width, height, point_x, point_y)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    if not (x <= point_x <= x + width and y <= point_y <= y + height):
        return None

    midpoint_x = x + width / 2.0
    midpoint_y = y + height / 2.0
    if point_x >= midpoint_x:
        return "NE" if point_y < midpoint_y else "SE"
    return "NO" if point_y < midpoint_y else "SO"


def resolve_vcp_scale_px(vcp_offset_mm: float, mm_per_px: float) -> float | None:
    offset = float(vcp_offset_mm)
    scale = float(mm_per_px)
    if not isfinite(offset) or not isfinite(scale) or scale <= 0.0 or not isfinite(offset / scale):
        return None
    return offset / scale


@dataclass(frozen=True)
class EstimatedMaskPose:
    x: float
    y: float
    angle_deg: float
    axis_vx: float
    axis_vy: float
    axis_quality: float
    mask_area_ratio: float
    mask_cut: bool


class MoldPoseEstimator:
    def __init__(self, border_margin_px: int = 3) -> None:
        self.border_margin_px = max(1, int(border_margin_px))

    def estimate(self, mask: np.ndarray) -> EstimatedMaskPose:
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2:
            raise ValueError("A máscara deve ter duas dimensões")
        ys, xs = np.nonzero(binary)
        if len(xs) < 3:
            raise ValueError("Máscara sem área suficiente")

        x = float(xs.mean())
        y = float(ys.mean())
        centered = np.column_stack((xs - x, ys - y)).astype(np.float64)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        major = eigenvectors[:, order[-1]]
        major_value = float(eigenvalues[order[-1]])
        minor_value = max(float(eigenvalues[order[-2]]), 1e-12)
        axis_quality = major_value / minor_value
        axis_vx, axis_vy = orient_north(float(major[0]), float(major[1]))

        margin = self.border_margin_px
        h, w = binary.shape
        mask_cut = bool(
            binary[:margin, :].any()
            or binary[max(0, h - margin) :, :].any()
            or binary[:, :margin].any()
            or binary[:, max(0, w - margin) :].any()
        )
        return EstimatedMaskPose(
            x=x,
            y=y,
            angle_deg=heading_north_deg(axis_vx, axis_vy),
            axis_vx=axis_vx,
            axis_vy=axis_vy,
            axis_quality=axis_quality,
            mask_area_ratio=float(binary.mean()),
            mask_cut=mask_cut,
        )


class VisionStabilityTracker:
    def __init__(
        self,
        samples: int,
        max_jitter_x_px: float,
        max_jitter_y_px: float,
        max_jitter_angle_deg: float,
    ) -> None:
        self.samples = max(2, int(samples))
        self.max_jitter_x_px = float(max_jitter_x_px)
        self.max_jitter_y_px = float(max_jitter_y_px)
        self.max_jitter_angle_deg = float(max_jitter_angle_deg)
        self._history: deque[tuple[float, float, float]] = deque(maxlen=self.samples)

    def clear(self) -> None:
        self._history.clear()

    def update(self, observation: VisionObservation) -> VisionObservation:
        if not all(observation.gates.values()):
            self.clear()
            return replace(
                observation,
                stable=False,
                sigma_x=0.0,
                sigma_y=0.0,
                sigma_angle_deg=0.0,
            )
        self._history.append((observation.x, observation.y, observation.angle_deg))
        if len(self._history) < self.samples:
            return replace(observation, stable=False)

        values = np.asarray(self._history, dtype=np.float64)
        sigma_x = float(np.std(values[:, 0]))
        sigma_y = float(np.std(values[:, 1]))
        reference = float(values[-1, 2])
        deltas = np.asarray(
            [directed_heading_delta(value, reference) for value in values[:, 2]],
            dtype=np.float64,
        )
        sigma_angle = float(sqrt(float(np.mean(np.square(deltas)))))
        stable = (
            sigma_x <= self.max_jitter_x_px
            and sigma_y <= self.max_jitter_y_px
            and sigma_angle <= self.max_jitter_angle_deg
        )
        return replace(
            observation,
            stable=stable,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_angle_deg=sigma_angle,
        )


def build_observation(
    pose: EstimatedMaskPose,
    confidence: float,
    instance_count: int,
    frame_shape: Iterable[int],
    quality: dict[str, float],
    vision_reference: dict[str, Any] | None = None,
) -> VisionObservation:
    height, width = list(frame_shape)[:2]
    reference = vision_reference or {}
    vcp_offset_mm = float(reference.get("vcp_offset_mm", 55.0))
    mm_per_px = float(reference.get("mm_per_px", 1.0))
    s_px = resolve_vcp_scale_px(vcp_offset_mm, mm_per_px)
    scale_ok = s_px is not None
    if scale_ok:
        vcpn_x, vcpn_y = vcpn_from_heading(pose.x, pose.y, pose.angle_deg, float(s_px))
    else:
        vcpn_x, vcpn_y = pose.x, pose.y
    roi_quadrant = (
        roi_quadrant_of_point(vcpn_x, vcpn_y, reference.get("roi_px"))
        if bool(reference.get("roi_enabled", False))
        else None
    )
    gates = {
        "single_instance": instance_count == 1,
        "confidence": confidence >= quality["min_confidence"],
        "mask_not_cut": not pose.mask_cut,
        "axis_quality": pose.axis_quality >= quality["min_axis_quality"],
        "mask_area": quality["min_mask_area_ratio"]
        <= pose.mask_area_ratio
        <= quality["max_mask_area_ratio"],
        "vcp_scale": scale_ok,
    }
    return VisionObservation(
        timestamp=utc_now(),
        x=pose.x,
        y=pose.y,
        angle_deg=pose.angle_deg,
        confidence=float(confidence),
        axis_quality=pose.axis_quality,
        mask_area_ratio=pose.mask_area_ratio,
        mask_cut=pose.mask_cut,
        instance_count=int(instance_count),
        stable=False,
        sigma_x=0.0,
        sigma_y=0.0,
        sigma_angle_deg=0.0,
        frame_width=int(width),
        frame_height=int(height),
        gates=gates,
        vcpn_x=vcpn_x,
        vcpn_y=vcpn_y,
        axis_ux=pose.axis_vx,
        axis_uy=pose.axis_vy,
        vcp_offset_mm=vcp_offset_mm,
        mm_per_px=mm_per_px,
        roi_quadrant=roi_quadrant,
    )
