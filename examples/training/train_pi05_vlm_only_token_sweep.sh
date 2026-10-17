#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_only_token_sweep.sh [base|finetuned|both] [SEED] [extra args...]
Defaults: both sources, seed 0, VLM_PROMPT_TOKEN_COUNTS="8 16 32 64".
Order: base-vlm8/16/32/64, then finetuned-vlm8/16/32/64 (eight independent runs).
Each run independently starts from its source; all runs are sequential.
GPU_IDS=0,1 and BATCH_SIZE=32 default to two-GPU DDP, global batch 64.
Only VLM prompts train; action prompts=0, backbone frozen. torch.compile is off by default.
Always train 3000 steps and save only 003000 (no initial or intermediate checkpoints).
W&B defaults to online / prompt-learning, with checkpoint artifacts disabled.

Input models, tokenizer and dataset default to /root/autodl-fs/wyn/.
OUTPUT_ROOT defaults to /root/autodl-tmp/chkpt/2601-lerobot/vlm-only
LOG_ROOT defaults to /root/autodl-tmp/logs/vlm-only
Run names: pi05-{base|finetuned}-vlm{COUNT}-seed{SEED}; RUN_PREFIX changes pi05.
Use BASE_PRETRAINED_PATH / FINETUNED_PRETRAINED_PATH / DATASET_ROOT / TOKENIZER_PATH
to override inputs. GPU_IDS, BATCH_SIZE, MAIN_PROCESS_PORT and DRY_RUN are supported.
Extra arguments follow train_pi05_vlm_only.sh, e.g. --wandb.* or --policy.optimizer_lr.
FLOW_STEPS and SAVE_ONLY_FINAL are fixed here. Use the single-run script to change them.
RUN_NAME, VLM_PROMPT_TOKENS and PRETRAINED_PATH must be unset for this launcher.
PROMPT_EXPORT_PATH exports initial prompt only, then exits (select exactly one source/token count).
PROMPT_EXPORT_REFERENCE optionally checks a previous swap's prompt_init.json before export.

All selected runs preflight before any training starts. Stop on the first failure.
Existing outputs are never overwritten or silently skipped. Select remaining sources
and token counts explicitly after a partial sweep, or choose a new RUN_PREFIX.

GPU_IDS=0,1 BATCH_SIZE=32 bash examples/training/train_pi05_vlm_only_token_sweep.sh both 0
DRY_RUN=true bash examples/training/train_pi05_vlm_only_token_sweep.sh both 0
VLM_PROMPT_TOKEN_COUNTS="32 64" bash examples/training/train_pi05_vlm_only_token_sweep.sh finetuned 0
EOF
  exit 0
fi

SELECT="${1:-both}"; SEED="${2:-0}"
if (( $# >= 2 )); then shift 2; elif (( $# )); then shift; fi
case "$SELECT" in
  both) SOURCES=(base finetuned) ;;
  base|finetuned) SOURCES=("$SELECT") ;;
  *) echo "Source must be base, finetuned or both" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid SEED" >&2; exit 2; fi
for key in RUN_NAME VLM_PROMPT_TOKENS PRETRAINED_PATH; do
  if [[ -n "${!key:-}" ]]; then echo "Sweep controls $key; see --help" >&2; exit 2; fi
done
read -r -a TOKEN_COUNTS <<< "${VLM_PROMPT_TOKEN_COUNTS-8 16 32 64}"
if (( ${#TOKEN_COUNTS[@]} == 0 )); then echo "VLM_PROMPT_TOKEN_COUNTS cannot be empty" >&2; exit 2; fi
if [[ -n "${PROMPT_EXPORT_PATH:-}" ]] && (( ${#SOURCES[@]} * ${#TOKEN_COUNTS[@]} != 1 )); then
  echo "PROMPT_EXPORT_PATH requires exactly one source and one token count" >&2; exit 2
fi
declare -A SEEN_COUNTS=()
for count in "${TOKEN_COUNTS[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid VLM token count: $count" >&2; exit 2; fi
  if [[ -n "${SEEN_COUNTS[$count]:-}" ]]; then echo "Duplicate VLM token count: $count" >&2; exit 2; fi
  SEEN_COUNTS["$count"]=1
done
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "Invalid DRY_RUN" >&2; exit 2; fi
RUN_PREFIX="${RUN_PREFIX:-pi05}"
export GPU_IDS="${GPU_IDS:-0,1}" BATCH_SIZE="${BATCH_SIZE:-32}"
export FLOW_STEPS=3000 SAVE_ONLY_FINAL=true
export BASE_PRETRAINED_PATH="${BASE_PRETRAINED_PATH:-/root/autodl-fs/wyn/models/pi05_libero_base}"
export FINETUNED_PRETRAINED_PATH="${FINETUNED_PRETRAINED_PATH:-/root/autodl-fs/wyn/models/pi05_libero_finetuned_v044}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-fs/wyn/models/google/paligemma-3b-pt-224}"
export DATASET_ROOT="${DATASET_ROOT:-/root/autodl-fs/wyn/datasets/libero}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/vlm-only}"
export LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/vlm-only}"
EXTRA_ARGS=(--policy.compile_model=false "$@")

run_one() {
  local source="$1" count="$2" dry="$3"
  VLM_PROMPT_TOKENS="$count" RUN_NAME="${RUN_PREFIX}-${source}-vlm${count}-seed${SEED}" DRY_RUN="$dry" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" "$source" "$SEED" "${EXTRA_ARGS[@]}"
}

echo "Checking all $((${#SOURCES[@]} * ${#TOKEN_COUNTS[@]})) runs: sources=${SOURCES[*]} tokens=${TOKEN_COUNTS[*]} seed=$SEED"
for source in "${SOURCES[@]}"; do
  for count in "${TOKEN_COUNTS[@]}"; do
    run_one "$source" "$count" true >/dev/null
  done
done
echo "All selected runs passed preflight; output=$OUTPUT_ROOT"
for source in "${SOURCES[@]}"; do
  for count in "${TOKEN_COUNTS[@]}"; do
    if [[ -n "${PROMPT_EXPORT_PATH:-}" ]]; then
      echo "Starting initialization export: source=$source VLM=$count seed=$SEED (0 updates)"
    else
      echo "Starting source=$source VLM=$count (3000 steps, final checkpoint only)"
    fi
    if run_one "$source" "$count" "$DRY_RUN"; then
      echo "Completed source=$source VLM=$count dry_run=$DRY_RUN"
    else
      status=$?
      echo "Sweep stopped: source=$source VLM=$count exit=$status; inspect $LOG_ROOT/${RUN_PREFIX}-${source}-vlm${count}-seed${SEED}.log" >&2
      exit "$status"
    fi
  done
done
