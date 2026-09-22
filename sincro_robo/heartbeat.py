from __future__ import annotations

from typing import Any

CIP_ECHO_ALARM = "CIP SEM ECO"


def echo_as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    if isinstance(value, (bytes, bytearray)):
        return bool(value[0]) if value else False
    return bool(value)


class HeartbeatToggle:
    """BOOL that flips on a fixed interval. First call is due immediately."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self.value = False
        self._last_toggle_mono: float | None = None

    def reset(self) -> None:
        self._last_toggle_mono = None

    def next_value(self, now: float) -> bool | None:
        if self._last_toggle_mono is not None and (now - self._last_toggle_mono) < self.interval_s:
            return None
        self.value = not self.value
        self._last_toggle_mono = now
        return self.value


class EchoWatchdog:
    """Healthy only after the first echo edge, then if the echo keeps changing."""

    def __init__(self, lost_after_s: float = 3.0) -> None:
        self.lost_after_s = float(lost_after_s)
        self._last_echo: bool | None = None
        self._last_change_mono: float | None = None
        self._seen_edge = False
        self._io_dead = False

    def reset_session(self) -> None:
        self._last_echo = None
        self._last_change_mono = None
        self._seen_edge = False
        self._io_dead = False

    def mark_io_error(self) -> None:
        self._io_dead = True

    def observe_echo(self, value: Any, now: float) -> None:
        self._io_dead = False
        bit = echo_as_bool(value)
        if self._last_echo is None:
            self._last_echo = bit
            return
        if bit != self._last_echo:
            self._last_echo = bit
            self._last_change_mono = now
            self._seen_edge = True

    @property
    def echo(self) -> bool | None:
        return self._last_echo

    def healthy(self, now: float) -> bool:
        if self._io_dead or not self._seen_edge or self._last_change_mono is None:
            return False
        return (now - self._last_change_mono) < self.lost_after_s
