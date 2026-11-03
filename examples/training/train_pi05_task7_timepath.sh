#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1}"
exec python -u -m lerobot.scripts.pi05_task7_timepath_experiment train "$@"
