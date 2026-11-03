#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_TASK7_WORK_DIR="${GRIPPER_TASK7_1200_WORK_DIR:-/data1/wyn/prompt-work/gripper-event-task7-1200}"
export GRIPPER_EVENT_DIR="$GRIPPER_TASK7_WORK_DIR/additive"
export GRIPPER_EVAL_DEFAULT_EPISODES=10
export GRIPPER_ONLY_SELECTED_TASK=true
# Never import an inherited noisy task8 protocol or old curriculum output root.
export GRIPPER_FIXED_NOISE_STD=0
export GRIPPER_FIXED_EVAL=false
export GRIPPER_EVAL_ROOT="${GRIPPER_TASK7_1200_EVAL_ROOT:-${OUTPUT_ROOT:-${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}}/eval/gripper-task7-1200}"
unset BASE_OUTPUT EVAL_LOG_ROOT LIBERO_FIXED_NOISE_CONFIG
unset LIBERO_ROBUSTNESS_MANIFEST LIBERO_BENCHMARK
# Append explicit controls after "$@" so optional inherited CLI flags cannot
# silently enable corruption or all-task evaluation.
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment eval "$@" \
    --only-selected-task --noise-std 0
