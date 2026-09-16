#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sincro_robo.config import load_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight read-only do SINCRO_ROBO no PCBOX")
    parser.add_argument("--config", default="config/pcbox.json")
    parser.add_argument("--check-network", action="store_true", help="Testa apenas TCP 44818 do CLP")
    args = parser.parse_args()
    config = load_config(args.config)
    results: dict[str, Any] = {
        "python": {
            "value": platform.python_version(),
            "ok": sys.version_info[:2] == (3, 12),
        },
        "architecture": {
            "value": platform.machine(),
            "ok": platform.machine().lower() in {"x86_64", "amd64"},
        },
        "sentech_profile": {
            "value": "/opt/sentech/.stprofile",
            "ok": Path("/opt/sentech/.stprofile").is_file(),
        },
    }
    for module in ("numpy", "cv2", "fastapi", "uvicorn", "rfdetr", "aphyt", "stapipy"):
        results[f"module:{module}"] = {"ok": importlib.util.find_spec(module) is not None}
    if results["module:rfdetr"]["ok"]:
        installed_rfdetr = importlib.metadata.version("rfdetr")
        expected_rfdetr = str(config["model"]["rfdetr_version"])
        results["rfdetr_version"] = {
            "value": installed_rfdetr,
            "expected": expected_rfdetr,
            "ok": installed_rfdetr == expected_rfdetr,
        }

    checkpoint = Path(config["model"]["checkpoint"])
    results["checkpoint"] = {
        "value": str(checkpoint),
        "ok": checkpoint.is_file(),
        "sha256": file_sha256(checkpoint) if checkpoint.is_file() else None,
    }
    plc_ip = os.environ.get("SINCRO_PLC_IP") or config["plc"].get("ip")
    results["plc_ip"] = {"value": plc_ip or "não definido", "ok": bool(plc_ip)}
    if args.check_network and plc_ip:
        try:
            with socket.create_connection((plc_ip, int(config["plc"]["port"])), timeout=3):
                pass
            results["plc_tcp"] = {"value": f"{plc_ip}:{config['plc']['port']}", "ok": True}
        except OSError as error:
            results["plc_tcp"] = {
                "value": f"{plc_ip}:{config['plc']['port']}",
                "ok": False,
                "error": str(error),
            }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item.get("ok", False) for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
