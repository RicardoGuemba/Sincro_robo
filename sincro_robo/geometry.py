from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import atan2, degrees, sqrt
from typing import Iterable

import numpy as np

from .domain import VisionObservation, utc_now


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected image axis to the continuous range [0, 180)."""
    normalized = float(angle_deg) % 180.0
    return 0.0 if abs(normalized - 180.0) < 1e-10 else normalized


def signed_axis_delta(angle_deg: float, reference_deg: float) -> float:
    """Smallest signed delta for axes where theta and theta + 180 are equal."""
    return ((float(angle_deg) - float(reference_deg) + 90.0) % 180.0) - 90.0


def axis_angle_from_vector(vx: float, vy: float) -> float:
    # Image coordinates grow downwards, so atan2(vy, vx) is clockwise-positive.
    return normalize_axis_angle(degrees(atan2(float(vy), float(vx))))


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
            angle_deg=axis_angle_from_vector(float(major[0]), float(major[1])),
            axis_vx=float(major[0]),
            axis_vy=float(major[1]),
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
            [signed_axis_delta(value, reference) for value in values[:, 2]],
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
) -> VisionObservation:
    height, width = list(frame_shape)[:2]
    gates = {
        "single_instance": instance_count == 1,
        "confidence": confidence >= quality["min_confidence"],
        "mask_not_cut": not pose.mask_cut,
        "axis_quality": pose.axis_quality >= quality["min_axis_quality"],
        "mask_area": quality["min_mask_area_ratio"]
        <= pose.mask_area_ratio
        <= quality["max_mask_area_ratio"],
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
    )

