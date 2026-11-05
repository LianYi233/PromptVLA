#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Three native closed-loop arms, one already trained Time-Input checkpoint.
# GPU_IDS defaults to 0,1; one GPU is also supported. No training is started.
exec python -u -m lerobot.scripts.pi05_task7_closed_loop "$@"
