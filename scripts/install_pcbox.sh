#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CHECKPOINT="buddmeyer_rfdetr_seg__seg_small__20260915_183353/checkpoint_best_total.pth"
if [[ ! -f "$CHECKPOINT" ]] || head -c 80 "$CHECKPOINT" | grep -q "git-lfs.github.com/spec/v1"; then
  echo "Checkpoint RF-DETR ausente. No clone: git lfs install && git lfs pull" >&2
  exit 2
fi

if [[ -z "${STAPIPY_WHEEL:-}" ]]; then
  echo "Defina STAPIPY_WHEEL com o caminho do wheel stapipy 1.2.3 cp312 linux_x86_64." >&2
  exit 2
fi

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[pcbox]'
.venv/bin/python -m pip install "$STAPIPY_WHEEL"

echo "Instalação concluída. Execute scripts/preflight.py antes de iniciar a câmera."

