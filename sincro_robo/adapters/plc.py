from __future__ import annotations

import threading
import time
from math import cos, degrees, radians, sin
from typing import Any, Callable

from ..domain import RobotPoseSnapshot, VisionObservation, utc_now


class SyntheticPoseReader:
    """Creates a correlated robot pose so local calibration can be exercised."""

    def __init__(
        self,
        vision_supplier: Callable[[], VisionObservation | None],
        plan_supplier: Callable[[], float | None],
        pick_offset_mm: tuple[float, float] = (57.5, 0.0),
    ) -> None:
        self.vision_supplier = vision_supplier
        self.plan_supplier = plan_supplier
        self.pick_offset_mm = pick_offset_mm
        self._opened = False

    def connect(self) -> None:
        self._opened = True

    def read_pose(self) -> RobotPoseSnapshot:
        if not self._opened:
            raise RuntimeError("PLC simulado não conectado")
        vision = self.vision_supplier()
        if vision is None:
            return RobotPoseSnapshot(utc_now(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
        # Ground-truth affine relationship for deterministic local validation.
        center_x = 110.0 + 0.54 * vision.x + 0.018 * vision.y
        center_y = 65.0 - 0.012 * vision.x + 0.49 * vision.y
        rz = (vision.angle_deg + 12.0) % 180.0
        dx, dy = self.pick_offset_mm
        angle = radians(rz)
        tcp_x = center_x + cos(angle) * dx - sin(angle) * dy
        tcp_y = center_y + sin(angle) * dx + cos(angle) * dy
        plan_z = self.plan_supplier()
        return RobotPoseSnapshot(
            timestamp=utc_now(),
            x=tcp_x,
            y=tcp_y,
            z=float(plan_z if plan_z is not None else 0.0),
            rx=180.0,
            ry=0.0,
            rz=rz,
            fresh=True,
        )

    def close(self) -> None:
        self._opened = False


class CipPoseReader:
    """Read-only Omron NX/NJ EtherNet/IP reader backed by aphyt."""

    def __init__(
        self,
        ip: str,
        tag: str,
        timeout_s: float = 10.0,
        angle_unit: str = "degrees",
    ) -> None:
        if not ip:
            raise ValueError("Defina SINCRO_PLC_IP ou plc.ip antes do modo PCBOX")
        self.ip = ip
        self.tag = tag
        self.timeout_s = float(timeout_s)
        if angle_unit not in {"degrees", "radians"}:
            raise ValueError("plc.angle_unit deve ser 'degrees' ou 'radians'")
        self.angle_unit = angle_unit
        self._plc: Any = None
        self._owner_thread: int | None = None

    def _assert_owner(self) -> None:
        if self._owner_thread is not None and self._owner_thread != threading.get_ident():
            raise RuntimeError("A sessão CIP deve permanecer na thread proprietária")

    def connect(self) -> None:
        self._owner_thread = threading.get_ident()
        from aphyt import omron  # type: ignore[import-not-found]

        plc = omron.n_series.NSeries()
        plc.connect_explicit(self.ip, connection_timeout=self.timeout_s)
        self._plc = plc

    def _read_array(self) -> list[float]:
        if self._plc is None:
            raise RuntimeError("Sessão CIP não conectada")
        value = self._plc.read_variable(self.tag)
        if isinstance(value, (list, tuple)) and len(value) >= 6:
            return [float(item) for item in value[:6]]
        # Some aphyt/Sysmac combinations expose array members individually.
        values = [self._plc.read_variable(f"{self.tag}[{index}]") for index in range(6)]
        return [float(item) for item in values]

    def read_pose(self) -> RobotPoseSnapshot:
        self._assert_owner()
        values = self._read_array()
        if len(values) != 6 or not all(isinstance(value, float) for value in values):
            raise ValueError(f"{self.tag}[0..5] não retornou seis valores numéricos")
        rx, ry, rz = values[3:6]
        if self.angle_unit == "radians":
            rx, ry, rz = degrees(rx), degrees(ry), degrees(rz)
        return RobotPoseSnapshot(
            timestamp=utc_now(),
            x=values[0],
            y=values[1],
            z=values[2],
            rx=rx,
            ry=ry,
            rz=rz,
            fresh=True,
        )

    def close(self) -> None:
        self._assert_owner()
        plc = self._plc
        self._plc = None
        try:
            if plc is not None:
                close = getattr(plc, "close_explicit", None) or getattr(plc, "close", None)
                if callable(close):
                    close()
        finally:
            self._owner_thread = None
