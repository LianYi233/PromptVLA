#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset BASE_OUTPUT EVAL_LOG_ROOT LIBERO_FIXED_NOISE_CONFIG
unset LIBERO_ROBUSTNESS_MANIFEST LIBERO_BENCHMARK LIBERO_SELECTED_TASK_ID
unset LIBERO_TASK7_OBJECT_SHIFT_CONFIG
exec python -u -m lerobot.scripts.pi05_task7_timepath_experiment eval "$@"
