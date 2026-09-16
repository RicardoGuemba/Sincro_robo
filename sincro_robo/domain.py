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
        result["prompt"] = "Gravar esta posição do robô?"
        return result

