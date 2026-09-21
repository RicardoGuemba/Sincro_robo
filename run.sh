#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Ambiente virtual ausente: $PYTHON" >&2
  echo "Crie com: python3.12 -m venv .venv && .venv/bin/python -m pip install -e '.[pcbox]'" >&2
  exit 2
fi

PROFILE="pcbox"
if [[ "${1:-}" == "pcbox" || "${1:-}" == "simulator" ]]; then
  PROFILE="$1"
  shift
fi

CONFIG="config/${PROFILE}.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "Configuração não encontrada: $CONFIG" >&2
  exit 2
fi

if [[ "$PROFILE" == "pcbox" ]]; then
  if [[ ! -f /opt/sentech/.stprofile ]]; then
    echo "Perfil do SentechSDK ausente: /opt/sentech/.stprofile" >&2
    exit 2
  fi
  if [[ -z "${SINCRO_PLC_IP:-}" ]]; then
    export SINCRO_PLC_IP=192.168.250.1
  fi
  set +u
  # shellcheck disable=SC1091
  source /opt/sentech/.stprofile
  set -u
fi

exec "$PYTHON" -m sincro_robo --config "$CONFIG" "$@"
