#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_EVENT_DIR="${GRIPPER_ADDITIVE_DIR:-/data1/wyn/prompt-work/gripper-event-additive-task8-r1}"
export GRIPPER_FIXED_NOISE_STD="${GRIPPER_FIXED_NOISE_STD:-0.10}"
export GRIPPER_FIXED_NOISE_SEED="${GRIPPER_FIXED_NOISE_SEED:-20261102}"
export GRIPPER_FIXED_NOISE_SCOPE="${GRIPPER_FIXED_NOISE_SCOPE:-task8}"
export GRIPPER_FIXED_EVAL=true
LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
IFS=, read -r -a gpus <<< "${GPU_IDS:-0,1,2,3}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "Need four GPU IDs via GPU_IDS=0,1,2,3" >&2
  exit 2
fi
declare -A used_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${used_gpus[$gpu]+x}" ]]; then
    echo "GPU IDs must be four distinct nonnegative integers" >&2
    exit 2
  fi
  used_gpus[$gpu]=1
done
declare -a cases=(uniform event uniform_time event_time)
declare -a pids=()
mkdir -p "$LOG_ROOT"
# Keep logs of different invocations independent; evaluation result identities
# themselves are based on the immutable, seed-specific noise configuration.
LOG_TAG="fixednoise-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
for i in 0 1 2 3; do
  case_name="${cases[$i]}"
  gpu="${gpus[$i]}"
  printf 'Launching %s on GPU %s; log=%s\n' "$case_name" "$gpu" \
    "$LOG_ROOT/eval-additive-${LOG_TAG}-${case_name}.log"
  GPU_ID="$gpu" bash "$ROOT/run_eval_libero_gripper_additive.sh" "$case_name" "$@" \
    > "$LOG_ROOT/eval-additive-${LOG_TAG}-${case_name}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for i in 0 1 2 3; do
  if ! wait "${pids[$i]}"; then
    echo "Evaluation failed: ${cases[$i]} (see $LOG_ROOT/eval-additive-${LOG_TAG}-${cases[$i]}.log)" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "All four evaluation preflights passed. No rollout or summary written."
  exit 0
fi
echo "All four evaluations exited successfully. Generating combined results."
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment summarize all "$@"
