#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1}"
# Uses both GPUs for matched task7 displacement eval, one or two model
# processes per phase. Loads old controls only if that seed wasn't tested.
# Prints a 5-model per-seed matrix and pooled report in the terminal.
exec python -u -m lerobot.scripts.pi05_task7_input_spectrum run "$@"
