#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_TASK7_WORK_DIR="${GRIPPER_TASK7_1200_WORK_DIR:-/data1/wyn/prompt-work/gripper-event-task7-1200}"
export GRIPPER_EVENT_DIR="$GRIPPER_TASK7_WORK_DIR/additive"
export GRIPPER_EXPERIMENT_DEFAULT_WANDB_MODE=offline
export GPU_IDS="${GPU_IDS:-1,3}"
# The focused script uses 1 task7 episode, 1000 uniform + 200 extra steps.
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment train "$@"
