from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class VisionObservation:
    timestamp: str
    x: float
    y: float
    angle_deg: float
    confidence: float
    axis_quality: float
    mask_area_ratio: float
    mask_cut: bool
    instance_count: int
    stable: bool
    sigma_x: float
    sigma_y: float
    sigma_angle_deg: float
    frame_width: int
    frame_height: int
    gates: dict[str, bool] = field(default_factory=dict)
    native_x: float | None = None
    native_y: float | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    reference_width: int | None = None
    reference_height: int | None = None

    def overlay_xy(self) -> tuple[float, float]:
        """Pixel coordinates on the original camera frame (mask centroid)."""
        return (
            self.x if self.native_x is None else float(self.native_x),
            self.y if self.native_y is None else float(self.native_y),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotPoseSnapshot:
    timestamp: str
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    fresh: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenVisionCapture:
    id: str
    session_id: str
    plan_z: float
    point_index: int
    role: str
    region: str
    vision: VisionObservation
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureCandidate:
    id: str
    session_id: str
    plan_z: float
    point_index: int
    role: str
    region: str
    vision: VisionObservation
    robot: RobotPoseSnapshot
    created_at: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prompt"] = "Ponto registrado"
        return result

