#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Offline complete denoising from shared pure noise using one Time-Input model.
# No training or MuJoCo execution. --dry-run prints the plan without a GPU.
if [[ -n "${GPU_ID:-}" ]]; then
  exec python -u -m lerobot.scripts.pi05_task7_full_sampling "$@" --gpu "$GPU_ID"
fi
exec python -u -m lerobot.scripts.pi05_task7_full_sampling "$@"
