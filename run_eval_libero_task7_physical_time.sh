#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Offline flow velocity MSE, no LIBERO simulation or image corruption.
unset LIBERO_FIXED_NOISE_CONFIG LIBERO_TASK7_OBJECT_SHIFT_CONFIG
unset LIBERO_BENCHMARK LIBERO_ROBUSTNESS_MANIFEST
exec python -u -m lerobot.scripts.pi05_task7_physical_time_experiment eval "$@"
