#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_TASK7_WORK_DIR="${GRIPPER_TASK7_1200_WORK_DIR:-/data1/wyn/prompt-work/gripper-event-task7-1200}"
export GRIPPER_EVENT_DIR="$GRIPPER_TASK7_WORK_DIR/additive"
export GRIPPER_EVAL_DEFAULT_EPISODES=10
export GRIPPER_ONLY_SELECTED_TASK=true
export GRIPPER_FIXED_NOISE_STD=0
export GRIPPER_FIXED_EVAL=false
export GRIPPER_EVAL_ROOT="${GRIPPER_TASK7_1200_EVAL_ROOT:-${OUTPUT_ROOT:-${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}}/eval/gripper-task7-1200}"
unset BASE_OUTPUT EVAL_LOG_ROOT LIBERO_FIXED_NOISE_CONFIG
unset LIBERO_ROBUSTNESS_MANIFEST LIBERO_BENCHMARK
IFS=, read -r -a gpus <<< "${GPU_IDS:-1,3}"
if [[ "${#gpus[@]}" -ne 2 || ! "${gpus[0]}" =~ ^[0-9]+$ \
        || ! "${gpus[1]}" =~ ^[0-9]+$ || "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "Use two distinct GPU indices, e.g. GPU_IDS=1,3" >&2
  exit 2
fi
if [[ "${DRY_RUN:-false}" != true ]]; then
  min_free="${MIN_FREE_GPU_MIB:-20000}"
  if [[ ! "$min_free" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GPU_MIB must be a nonnegative integer" >&2
    exit 2
  fi
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found" >&2; exit 2; }
  for gpu in "${gpus[@]}"; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free \
      --format=csv,noheader,nounits) || exit 2
    free_mib="${free_mib//[[:space:]]/}"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
      echo "Cannot read free GPU memory for GPU $gpu" >&2
      exit 2
    fi
    echo "GPU $gpu: ${free_mib} MiB free (minimum ${min_free})"
    if (( free_mib < min_free )); then
      echo "Insufficient GPU $gpu memory; no evaluations launched" >&2
      exit 1
    fi
  done
fi
LOG_DIR="${LOG_DIR:-/data1/wyn/chkpt/2601-lerobot/logs/gripper-task7-1200}"
mkdir -p "$LOG_DIR"
LOG_TAG="$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
run_group() {
  local gpu="$1"
  shift
  local case_name
  for case_name in "$@"; do
    echo "GPU $gpu: evaluating $case_name; task7 only, 10 episodes, clean" >&2
    GPU_ID="$gpu" bash "$ROOT/run_eval_libero_gripper_task7.sh" "$case_name" \
      "${GLOBAL_ARGS[@]}" >"$LOG_DIR/${LOG_TAG}-${case_name}.log" 2>&1 || return 1
  done
}
declare -a GLOBAL_ARGS=("$@")
run_group "${gpus[0]}" uniform event &
p1="$!"
run_group "${gpus[1]}" uniform_time event_time &
p2="$!"
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "At least one task7 evaluation failed. Inspect $LOG_DIR/${LOG_TAG}-*.log" >&2
  exit 1
fi
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "Task7 dry-run preflight completed; no episodes simulated"
  exit 0
fi
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment summarize all "$@" \
  --only-selected-task --noise-std 0
