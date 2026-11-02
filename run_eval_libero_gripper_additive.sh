#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_EVENT_DIR="${GRIPPER_ADDITIVE_DIR:-/data1/wyn/prompt-work/gripper-event-additive-task8-r1}"
# Noise is evaluation-only: same protocol/seed for all checkpoints.
export GRIPPER_FIXED_NOISE_STD="${GRIPPER_FIXED_NOISE_STD:-0.10}"
export GRIPPER_FIXED_NOISE_SEED="${GRIPPER_FIXED_NOISE_SEED:-20261102}"
export GRIPPER_FIXED_NOISE_SCOPE="${GRIPPER_FIXED_NOISE_SCOPE:-task8}"
export GRIPPER_FIXED_EVAL=true
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment eval "$@"
