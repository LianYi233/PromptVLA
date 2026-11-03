#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset BASE_OUTPUT EVAL_LOG_ROOT LIBERO_FIXED_NOISE_CONFIG
unset LIBERO_ROBUSTNESS_MANIFEST LIBERO_BENCHMARK LIBERO_SELECTED_TASK_ID
unset LIBERO_TASK7_OBJECT_SHIFT_CONFIG
IFS=, read -r -a gpus <<< "${GPU_IDS:-0,1}"
if [[ "${#gpus[@]}" -ne 2 || ! "${gpus[0]}" =~ ^[0-9]+$ \
    || ! "${gpus[1]}" =~ ^[0-9]+$ || "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "Set two distinct GPUs, e.g. GPU_IDS=0,1" >&2
  exit 2
fi
min_free="${MIN_FREE_GPU_MIB:-20000}"
if [[ "${DRY_RUN:-false}" != true ]]; then
  if [[ ! "$min_free" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GPU_MIB must be a nonnegative integer" >&2
    exit 2
  fi
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
  for gpu in "${gpus[@]}"; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    free_mib="${free_mib//[[:space:]]/}"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < min_free )); then
      echo "GPU $gpu has ${free_mib} MiB free, needs $min_free MiB" >&2
      exit 1
    fi
    echo "GPU $gpu: $free_mib MiB free"
  done
fi
log_dir="${LOG_DIR:-/data1/wyn/chkpt/2601-lerobot/logs/task7-timepath-qkv}"
mkdir -p "$log_dir"
tag="$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
args=("$@")
run_pair() {
  local left="$1" right="$2" phase="$3"
  echo "$phase: GPU ${gpus[0]} -> $left, GPU ${gpus[1]} -> $right"
  local log_left="$log_dir/${tag}-${left}.log"
  local log_right="$log_dir/${tag}-${right}.log"
  echo "Logs: $log_left / $log_right"
  GPU_ID="${gpus[0]}" bash "$ROOT/run_eval_libero_task7_timepath.sh" "$left" "${args[@]}" \
    >"$log_left" 2>&1 &
  local p0=$!
  GPU_ID="${gpus[1]}" bash "$ROOT/run_eval_libero_task7_timepath.sh" "$right" "${args[@]}" \
    >"$log_right" 2>&1 &
  local p1=$!
  local err=0
  wait "$p0" || err=1
  wait "$p1" || err=1
  if (( err )); then
    echo "$phase failed; inspect $log_left and $log_right" >&2
    return 1
  fi
}
run_pair rope time_qk "Phase 1"
run_pair time_v time_input "Phase 2"
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "Dry run complete; no environments were simulated"
  exit 0
fi
exec python -u -m lerobot.scripts.pi05_task7_timepath_experiment summarize all "${args[@]}"
