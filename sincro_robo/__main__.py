from __future__ import annotations

import argparse

import uvicorn

from .api import create_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SINCRO_ROBO")
    parser.add_argument("--config", default=None, help="Caminho do arquivo JSON de configuração")
    parser.add_argument("--host", default=None, help="Interface HTTP")
    parser.add_argument("--port", type=int, default=None, help="Porta HTTP")
    args = parser.parse_args()
    config = load_config(args.config)
    host = args.host or config["app"]["host"]
    port = args.port or int(config["app"]["port"])
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

