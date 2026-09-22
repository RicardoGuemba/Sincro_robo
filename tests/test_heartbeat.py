from __future__ import annotations

from sincro_robo.heartbeat import EchoWatchdog, HeartbeatToggle, echo_as_bool


def test_echo_as_bool_accepts_common_plc_shapes() -> None:
    assert echo_as_bool(True) is True
    assert echo_as_bool(0) is False
    assert echo_as_bool("true") is True
    assert echo_as_bool(b"\x01") is True
    assert echo_as_bool(b"") is False


def test_toggle_is_due_immediately_then_respects_interval() -> None:
    toggle = HeartbeatToggle(interval_s=1.0)
    assert toggle.next_value(0.0) is True
    assert toggle.next_value(0.9) is None
    assert toggle.next_value(1.0) is False
    assert toggle.next_value(1.99) is None
    assert toggle.next_value(2.0) is True


def test_watchdog_exposes_last_echo_bit() -> None:
    watchdog = EchoWatchdog(lost_after_s=3.0)
    assert watchdog.echo is None
    watchdog.observe_echo(False, 0.0)
    assert watchdog.echo is False
    watchdog.observe_echo(True, 1.0)
    assert watchdog.echo is True


def test_watchdog_blocks_until_first_echo_edge() -> None:
    watchdog = EchoWatchdog(lost_after_s=3.0)
    watchdog.observe_echo(False, 0.0)
    assert watchdog.healthy(0.5) is False
    watchdog.observe_echo(False, 2.9)
    assert watchdog.healthy(2.9) is False
    watchdog.observe_echo(True, 1.0)
    assert watchdog.healthy(1.0) is True


def test_watchdog_dies_after_lost_window_and_recovers_on_next_edge() -> None:
    watchdog = EchoWatchdog(lost_after_s=3.0)
    watchdog.observe_echo(False, 0.0)
    watchdog.observe_echo(True, 1.0)
    assert watchdog.healthy(3.9) is True
    assert watchdog.healthy(4.0) is False
    watchdog.observe_echo(False, 5.0)
    assert watchdog.healthy(5.0) is True


def test_io_error_is_dead_until_a_new_session_sees_an_edge() -> None:
    watchdog = EchoWatchdog(lost_after_s=3.0)
    watchdog.observe_echo(False, 0.0)
    watchdog.observe_echo(True, 1.0)
    watchdog.mark_io_error()
    assert watchdog.healthy(1.1) is False
    watchdog.reset_session()
    watchdog.observe_echo(True, 10.0)
    assert watchdog.healthy(10.0) is False
    watchdog.observe_echo(False, 11.0)
    assert watchdog.healthy(11.0) is True
