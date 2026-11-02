#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_EVENT_DIR="${GRIPPER_ADDITIVE_DIR:-/data1/wyn/prompt-work/gripper-event-additive-task8-r1}"
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment eval "$@"
