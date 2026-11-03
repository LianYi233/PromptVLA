#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRIPPER_EVENT_DIR="${GRIPPER_TASK7_1200_WORK_DIR:-/data1/wyn/prompt-work/gripper-event-task7-1200}/additive"
export GRIPPER_ONLY_SELECTED_TASK=true GRIPPER_EVAL_DEFAULT_EPISODES=10
export GRIPPER_FIXED_NOISE_STD=0 GRIPPER_FIXED_EVAL=false
export GRIPPER_EVAL_ROOT="${GRIPPER_TASK7_SHIFT_EVAL_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-task7-object-shift}"
unset BASE_OUTPUT EVAL_LOG_ROOT LIBERO_FIXED_NOISE_CONFIG
unset LIBERO_ROBUSTNESS_MANIFEST LIBERO_BENCHMARK
IFS=, read -r -a gpus <<< "${GPU_IDS:-1,3}"
if [[ "${#gpus[@]}" -ne 2 || ! "${gpus[0]}" =~ ^[0-9]+$ \
    || ! "${gpus[1]}" =~ ^[0-9]+$ || "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "Choose two distinct GPU IDs, e.g. GPU_IDS=1,3" >&2
  exit 2
fi
ARGS=("$@")
has_shift=false
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == --shift-m || "$arg" == --shift-m=* ]]; then
    has_shift=true
  fi
done
if [[ "$has_shift" == false ]]; then
  # Never let a stale TASK7_OBJECT_SHIFT_M from another shell override the
  # preregistered Uniform-only calibration. Explicit --shift-m is the override.
  unset TASK7_OBJECT_SHIFT_M
  CALIBRATION="${TASK7_SHIFT_CALIBRATION:-/data1/wyn/prompt-work/gripper-event-task7-1200/shift-calibration.json}"
  if [[ ! -f "$CALIBRATION" ]]; then
    echo "Missing calibrated shift: $CALIBRATION" >&2
    echo "Run python -m lerobot.scripts.pi05_task7_shift_calibrate calibrate first, or pass --shift-m 0.04" >&2
    exit 2
  fi
  TASK7_OBJECT_SHIFT_M=$(
    python -m lerobot.scripts.pi05_task7_shift_calibrate selected --file "$CALIBRATION"
  )
  export TASK7_OBJECT_SHIFT_M
fi
min_free="${MIN_FREE_GPU_MIB:-20000}"
if [[ "${DRY_RUN:-false}" != true ]]; then
  if [[ ! "$min_free" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GPU_MIB must be a nonnegative integer" >&2
    exit 2
  fi
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 2; }
  for gpu in "${gpus[@]}"; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    free="${free//[[:space:]]/}"
    if [[ ! "$free" =~ ^[0-9]+$ || "$free" -lt "$min_free" ]]; then
      echo "GPU $gpu insufficient free memory: $free MiB (minimum $min_free)" >&2
      exit 1
    fi
    echo "GPU $gpu: $free MiB free"
  done
fi
LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
log_dir="$LOG_ROOT/gripper-task7-object-shift"
mkdir -p "$log_dir"
tag="$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
echo "Task7 displacement: ${TASK7_OBJECT_SHIFT_M:-from explicit --shift-m}, 2 GPUs, 10 episodes/task"
run_pair() {
  local label="$1" first="$2" second="$3"
  local one_log="$log_dir/${tag}-${first}.log"
  local two_log="$log_dir/${tag}-${second}.log"
  echo "$label: GPU ${gpus[0]} -> $first; GPU ${gpus[1]} -> $second"
  echo "Logs: $one_log ; $two_log"
  GPU_ID="${gpus[0]}" bash "$ROOT/run_eval_libero_gripper_task7_shift.sh" "$first" "${ARGS[@]}" \
    >"$one_log" 2>&1 &
  local p1=$!
  GPU_ID="${gpus[1]}" bash "$ROOT/run_eval_libero_gripper_task7_shift.sh" "$second" "${ARGS[@]}" \
    >"$two_log" 2>&1 &
  local p2=$!
  local failed=0
  wait "$p1" || failed=1
  wait "$p2" || failed=1
  if (( failed )); then
    echo "$label failed; inspect $one_log and $two_log" >&2
    return 1
  fi
}
run_pair "Phase 1" uniform event
run_pair "Phase 2" uniform_time event_time
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "Dry-run passed; no rollout or summary generated"
  exit 0
fi
exec python -u -m lerobot.scripts.pi05_gripper_event_experiment summarize all "${ARGS[@]}" \
  --only-selected-task --noise-std 0
