#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_only_pair.sh [SEED] [common extra train args...]
TRAIN_LAYOUT=parallel (default): GPU 0 -> base, GPU 1 -> finetuned; per-GPU batch 32.
TRAIN_LAYOUT=ddp: two GPUs jointly train base, then jointly train finetuned; per-GPU batch 16.
Both defaults keep global batch 32. GPU_IDS must list exactly two distinct GPUs.
Use BATCH_SIZE to override PER-GPU batch. Both start independently from their source weights.
Both preflight before launching. Separate ports, runs, logs and compile caches.
In parallel mode a failed run does not cancel the independent healthy run; wait for both,
then return failure if either failed. Ctrl-C/TERM cleans up this launcher's process groups.
MAIN_PROCESS_PORT defaults to 29531 (the second source uses the next port).
Other settings are inherited from train_pi05_vlm_only.sh. RUN_NAME, PRETRAINED_PATH and
NUM_PROCESSES are controlled here; use BASE_PRETRAINED_PATH / FINETUNED_PRETRAINED_PATH.
EOF
  exit 0
fi
SEED="${1:-0}"; if (( $# )); then shift; fi
for key in RUN_NAME PRETRAINED_PATH NUM_PROCESSES; do
  if [[ -n "${!key:-}" ]]; then echo "Pair launcher controls $key; see --help" >&2; exit 2; fi
done
TRAIN_LAYOUT="${TRAIN_LAYOUT:-parallel}"
GPU_IDS="${GPU_IDS:-0,1}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+,[0-9]+$ ]]; then echo "GPU_IDS must list exactly two GPUs" >&2; exit 2; fi
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then echo "GPU IDs must be distinct" >&2; exit 2; fi
PORT="${MAIN_PROCESS_PORT:-29531}"
if [[ ! "$PORT" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65534 )); then echo "Invalid MAIN_PROCESS_PORT" >&2; exit 2; fi
case "$TRAIN_LAYOUT" in
  parallel) export BATCH_SIZE="${BATCH_SIZE:-32}" ;;
  ddp) export BATCH_SIZE="${BATCH_SIZE:-16}" ;;
  *) echo "TRAIN_LAYOUT must be parallel or ddp" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "Invalid DRY_RUN" >&2; exit 2; fi
EXTRA_ARGS=("$@")
run_one() {
  local source="$1" index="$2" dry="$3" ids="$GPU_IDS"
  if [[ "$TRAIN_LAYOUT" == parallel ]]; then ids="${GPUS[$index]}"; fi
  GPU_IDS="$ids" MAIN_PROCESS_PORT="$((PORT + index))" DRY_RUN="$dry" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" "$source" "$SEED" "${EXTRA_ARGS[@]}"
}
run_one base 0 true >/dev/null
run_one finetuned 1 true >/dev/null
echo "Both sources passed preflight; layout=$TRAIN_LAYOUT GPUs=$GPU_IDS"
if [[ "$DRY_RUN" == true || "$TRAIN_LAYOUT" == ddp ]]; then
  run_one base 0 "$DRY_RUN"
  run_one finetuned 1 "$DRY_RUN"
  exit 0
fi
command -v setsid >/dev/null
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
for index in 0 1; do
  source=base; if (( index == 1 )); then source=finetuned; fi
  GPU_IDS="${GPUS[$index]}" MAIN_PROCESS_PORT="$((PORT + index))" DRY_RUN=false \
    setsid bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" "$source" "$SEED" "${EXTRA_ARGS[@]}" &
  PIDS+=("$!")
done
RESULT=0
for index in 0 1; do
  if wait "${PIDS[$index]}"; then
    echo "Source index $index completed successfully"
  else
    status=$?
    echo "Source index $index failed with exit code $status; inspect its separate training log" >&2
    RESULT=1
  fi
done
exit "$RESULT"
