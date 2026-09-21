from __future__ import annotations

import numpy as np
import pytest

from sincro_robo.adapters.camera import limit_frame_resolution


def test_sentech_4_3_downscales_to_960x720() -> None:
    frame = np.zeros((1944, 2592, 3), dtype=np.uint8)
    out = limit_frame_resolution(frame, 960)
    assert out.shape[:2] == (720, 960)


def test_already_960x720_is_unchanged() -> None:
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    out = limit_frame_resolution(frame, 960)
    assert out is frame
    assert out.shape[:2] == (720, 960)


def test_smaller_4_3_frame_is_not_upscaled() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    out = limit_frame_resolution(frame, 960)
    assert out is frame
    assert out.shape[:2] == (600, 800)


def test_max_dimension_zero_skips_resize() -> None:
    frame = np.zeros((1944, 2592, 3), dtype=np.uint8)
    out = limit_frame_resolution(frame, 0)
    assert out is frame
    assert out.shape[:2] == (1944, 2592)


def test_config_defaults_are_960_by_4_3() -> None:
    from sincro_robo.config import DEFAULT_CONFIG, load_config

    assert DEFAULT_CONFIG["camera"]["max_dimension"] == 960
    assert DEFAULT_CONFIG["camera"]["width"] == 960
    assert DEFAULT_CONFIG["camera"]["height"] == 720
    pcbox = load_config("config/pcbox.json")
    assert pcbox["camera"]["max_dimension"] == 960
    assert pcbox["pixel_reference"]["destination_width"] == 960
    assert pcbox["pixel_reference"]["destination_height"] == 720
