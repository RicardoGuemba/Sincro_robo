from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import Any, Iterable, Optional

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


def configured_pixel_scales(pixel_reference: dict[str, Any]) -> tuple[float, float]:
    source_width = int(pixel_reference["source_width"])
    source_height = int(pixel_reference["source_height"])
    destination_width = int(pixel_reference["destination_width"])
    destination_height = int(pixel_reference["destination_height"])
    if min(source_width, source_height, destination_width, destination_height) <= 0:
        raise ValueError("pixel_reference deve ter resoluções positivas")
    return destination_width / source_width, destination_height / source_height


def reference_scale_factors(
    frame_width: int,
    frame_height: int,
    pixel_reference: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Map native mask centroid pixels into the pick-and-place frame, once.

    Identity when the frame is already the destination (e.g. 960×720) or when it
    does not match the configured source resolution.
    """
    width = int(frame_width)
    height = int(frame_height)
    if width <= 0 or height <= 0:
        raise ValueError("Resolução do frame inválida")
    ref = pixel_reference or {}
    source_width = int(ref.get("source_width", width))
    source_height = int(ref.get("source_height", height))
    destination_width = int(ref.get("destination_width", width))
    destination_height = int(ref.get("destination_height", height))
    if min(source_width, source_height, destination_width, destination_height) <= 0:
        raise ValueError("pixel_reference deve ter resoluções positivas")
    if width == destination_width and height == destination_height:
        return 1.0, 1.0
    if width == source_width and height == source_height:
        return destination_width / source_width, destination_height / source_height
    return 1.0, 1.0


def map_to_reference(native_x: float, native_y: float, scale_x: float, scale_y: float) -> tuple[float, float]:
    return float(native_x) * float(scale_x), float(native_y) * float(scale_y)


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
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0


def _angle_and_sides_from_min_area_rect(
    bool_mask: np.ndarray,
) -> Optional[tuple[float, float, float]]:
    """minAreaRect on the largest contour → (angle_deg, major_len, minor_len)."""
    import cv2

    bin_mask = np.asarray(bool_mask, dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None
    (_centre, (width, height), angle_rect) = cv2.minAreaRect(contour)
    width = float(width)
    height = float(height)
    if width <= 0 and height <= 0:
        return None
    if width >= height:
        major_len, minor_len = width, height
        angle = float(angle_rect)
    else:
        major_len, minor_len = height, width
        angle = float(angle_rect) + 90.0
    return normalize_axis_angle(angle), major_len, max(minor_len, 1.0)


def _angle_and_sides_from_pca(
    xs: np.ndarray,
    ys: np.ndarray,
    mean_x: float,
    mean_y: float,
) -> tuple[float, float, float]:
    """Fallback PCA: angle in [0, 180), side lengths ≈ 4·sqrt(λ)."""
    dx = xs.astype(np.float64) - mean_x
    dy = ys.astype(np.float64) - mean_y
    cov_xx = float(np.mean(dx * dx))
    cov_yy = float(np.mean(dy * dy))
    cov_xy = float(np.mean(dx * dy))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, 1.0
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major_value = max(float(eigenvalues[0]), 0.0)
    minor_value = max(float(eigenvalues[1]), 0.0)
    major_vec = eigenvectors[:, 0]
    angle = axis_angle_from_vector(float(major_vec[0]), float(major_vec[1]))
    major_len = 4.0 * float(np.sqrt(major_value)) if major_value > 0 else 1.0
    minor_len = 4.0 * float(np.sqrt(minor_value)) if minor_value > 0 else 1.0
    return angle, major_len, minor_len


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
        rect = _angle_and_sides_from_min_area_rect(binary)
        if rect is not None:
            angle_deg, major_len, minor_len = rect
        else:
            angle_deg, major_len, minor_len = _angle_and_sides_from_pca(xs, ys, x, y)
        axis_quality = float(major_len / minor_len) if minor_len > 1e-9 else 1.0

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
            angle_deg=angle_deg,
            axis_vx=cos(radians(angle_deg)),
            axis_vy=sin(radians(angle_deg)),
            axis_quality=axis_quality,
            mask_area_ratio=float(binary.mean()),
            mask_cut=mask_cut,
            major_axis_length=float(major_len),
            minor_axis_length=float(minor_len),
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
        overlay_x, overlay_y = observation.overlay_xy()
        self._history.append((overlay_x, overlay_y, observation.angle_deg))
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
    pixel_reference: dict[str, Any] | None = None,
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
    scale_x, scale_y = reference_scale_factors(int(width), int(height), pixel_reference)
    mapped_x, mapped_y = map_to_reference(pose.x, pose.y, scale_x, scale_y)
    ref = pixel_reference or {}
    return VisionObservation(
        timestamp=utc_now(),
        x=mapped_x,
        y=mapped_y,
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
        native_x=pose.x,
        native_y=pose.y,
        scale_x=scale_x,
        scale_y=scale_y,
        reference_width=int(ref.get("destination_width", width)),
        reference_height=int(ref.get("destination_height", height)),
    )

