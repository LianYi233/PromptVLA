#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1}"
# THREE NEW time-input variants; old 1000-step RoPE and original Time-Input
# checkpoints are reused verbatim and are NEVER trained by this launcher.
exec python -u -m lerobot.scripts.pi05_task7_input_spectrum train "$@"
