from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "SINCRO_ROBO",
        "host": "127.0.0.1",
        "port": 8080,
        "data_dir": "data",
        "database": "data/sincro_robo.sqlite3",
        "exports_dir": "data/exports",
    },
    "camera": {
        "provider": "synthetic",
        "device_index": 0,
        "fetch_timeout_ms": 400,
        "target_fps": 8.0,
        "width": 960,
        "height": 540,
    },
    "model": {
        "provider": "synthetic",
        "bundle_dir": "buddmeyer_rfdetr_seg__seg_small__20260915_183353",
        "checkpoint": "buddmeyer_rfdetr_seg__seg_small__20260915_183353/checkpoint_best_total.pth",
        "threshold": 0.3,
        "resolution": 384,
        "class_name": "sku",
        "device": "auto",
        "rfdetr_version": "1.10.1",
    },
    "plc": {
        "provider": "synthetic",
        "ip": "",
        "port": 44818,
        "connection_timeout_s": 10.0,
        "poll_interval_s": 0.2,
        "freshness_timeout_s": 1.5,
        "pose_tag": "RobFrom_Coord_CurrBase_Tool",
        "angle_unit": "degrees",
    },
    "vision_quality": {
        "min_confidence": 0.3,
        "min_axis_quality": 1.35,
        "border_margin_px": 3,
        "min_mask_area_ratio": 0.002,
        "max_mask_area_ratio": 0.85,
        "stability_samples": 6,
        "max_jitter_x_px": 2.5,
        "max_jitter_y_px": 2.5,
        "max_jitter_angle_deg": 1.5,
    },
    "calibration": {
        "default_planes_mm": [0.0, 200.0, 400.0],
        "default_points_per_plane": 7,
        "max_points_per_plane": 9,
        "adjustment_points_per_plane": 5,
        "validation_points_per_plane": 2,
        "pick_offset_local_mm": [57.5, 0.0],
        "xy_tolerance_mm": 20.0,
        "angular_tolerance_deg": 5.0,
        "duplicate_distance_px": 12.0,
        "duplicate_angle_deg": 3.0,
        "plan_z_tolerance_mm": 20.0,
        "target_region_radius_norm": 0.24,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (root / path).resolve())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("SINCRO_CONFIG", "config/simulator.json"))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    overrides: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            overrides = json.load(stream)
    config = _deep_merge(DEFAULT_CONFIG, overrides)

    plc_ip = os.environ.get("SINCRO_PLC_IP")
    if plc_ip:
        config["plc"]["ip"] = plc_ip

    for section, key in (
        ("app", "data_dir"),
        ("app", "database"),
        ("app", "exports_dir"),
        ("model", "bundle_dir"),
        ("model", "checkpoint"),
    ):
        config[section][key] = _resolve_path(config[section][key], PROJECT_ROOT)
    config["_config_path"] = str(config_path.resolve())
    return config
