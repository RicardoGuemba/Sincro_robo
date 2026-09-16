#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f /opt/sentech/.stprofile ]]; then
  echo "Perfil do SentechSDK ausente: /opt/sentech/.stprofile" >&2
  exit 2
fi
if [[ -z "${SINCRO_PLC_IP:-}" ]]; then
  echo "Defina SINCRO_PLC_IP com o IP confirmado do NX102." >&2
  exit 2
fi

set +u
source /opt/sentech/.stprofile
set -u

exec .venv/bin/python -m sincro_robo --config config/pcbox.json

