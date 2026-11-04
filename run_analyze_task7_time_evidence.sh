#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Default --mode success is CPU-only. --mode probe/all runs finite offline
# velocity diagnostics on an explicit GPU, no MuJoCo action execution.
if [[ -n "${GPU_ID:-}" ]]; then
  exec python -u -m lerobot.scripts.pi05_task7_time_evidence "$@" --gpu "$GPU_ID"
fi
exec python -u -m lerobot.scripts.pi05_task7_time_evidence "$@"
