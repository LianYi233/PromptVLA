#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=, read -r -a gpus <<< "${GPU_IDS:-0,1}"
if [[ "${#gpus[@]}" -ne 2 || ! "${gpus[0]}" =~ ^[0-9]+$ \
  || ! "${gpus[1]}" =~ ^[0-9]+$ || "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "Choose two distinct GPU indices, e.g. GPU_IDS=0,1" >&2
  exit 2
fi
if [[ "${DRY_RUN:-false}" != true ]]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing" >&2; exit 2; }
  min_mib="${MIN_FREE_GPU_MIB:-20000}"
  if [[ ! "$min_mib" =~ ^[0-9]+$ ]]; then echo "Invalid memory threshold" >&2; exit 2; fi
  for gpu in "${gpus[@]}"; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    free="${free//[[:space:]]/}"
    if [[ ! "$free" =~ ^[0-9]+$ ]] || (( free < min_mib )); then
      echo "GPU $gpu: $free MiB free, need $min_mib MiB" >&2
      exit 1
    fi
  done
fi
log_dir="${CADENCE_EVAL_LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs/task7-physical-time}"
mkdir -p "$log_dir"
tag="$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
args=("$@")
phase() {
  local case0="$1" case1="$2"
  echo "GPU ${gpus[0]}: $case0 | GPU ${gpus[1]}: $case1"
  GPU_ID="${gpus[0]}" bash "$ROOT/run_eval_libero_task7_physical_time.sh" \
    "$case0" "${args[@]}" --gpu "${gpus[0]}" >"$log_dir/${tag}-$case0.log" 2>&1 &
  local p0=$!
  GPU_ID="${gpus[1]}" bash "$ROOT/run_eval_libero_task7_physical_time.sh" \
    "$case1" "${args[@]}" --gpu "${gpus[1]}" >"$log_dir/${tag}-$case1.log" 2>&1 &
  local p1=$!
  local failed=0
  wait "$p0" || failed=1
  wait "$p1" || failed=1
  if (( failed )); then
    echo "Offline eval failed. Logs: $log_dir/${tag}-*.log" >&2
    return 1
  fi
}
phase rope index
phase global_dt physical
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "Dry run completed; no forward passes"
  exit 0
fi
exec python -u -m lerobot.scripts.pi05_task7_physical_time_experiment summarize all "${args[@]}"
