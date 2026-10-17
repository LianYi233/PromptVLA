#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_vlm_token_sweep.sh [all|no10|SUITE] [TRAINING_SEED]

Evaluate vlm_only checkpoints with 1, 2, 8, 16 VLM prompt tokens sequentially.
Default: checkpoint 003000, all four LIBERO suites, 10 episodes/task, GPU 0.
Each model uses separate output directories, logs and task-level resume records.

GPU_ID=0 bash run_eval_libero_vlm_token_sweep.sh all 0
DRY_RUN=true bash run_eval_libero_vlm_token_sweep.sh all 0
VLM_PROMPT_TOKEN_COUNTS="8 16" bash run_eval_libero_vlm_token_sweep.sh libero_10 0

CKPT_ROOT defaults to /root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/vlm-token-sweep
BASE_OUTPUT defaults to /root/autodl-tmp/eval/prompt-learning/vlm-token-sweep
RUN_PREFIX defaults to pi05-vlm_only; names are PREFIX-vlmN-seedS.
Other overrides: GPU_ID, CHECKPOINT_STEP, EPISODES_PER_TASK, LIBERO_CONFIG_PATH.
Token counts are validated against each checkpoint's saved config.json before
starting any evaluation. Missing checkpoints/mismatches stop the whole sweep.
Evaluation stops on the first failure; rerun the same command to resume saved tasks.
SEED selects the training run. The evaluator's default environment/global seed
remains 1000, shared across these models.
EOF
  exit 0
fi
if (( $# > 2 )); then
  echo "Expected [all|no10|SUITE] [TRAINING_SEED]; use --help" >&2
  exit 2
fi
MODE="${1:-all}"
SEED="${2:-0}"
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then
  echo "DRY_RUN must be true or false" >&2
  exit 2
fi
if [[ -n "${BASE_CKPT:-}" || -n "${RUN_NAME:-}" || -n "${VLM_PROMPT_TOKENS:-}" ]]; then
  echo "For sweeps, use CKPT_ROOT, RUN_PREFIX and VLM_PROMPT_TOKEN_COUNTS; unset BASE_CKPT/RUN_NAME/VLM_PROMPT_TOKENS" >&2
  exit 2
fi
read -r -a TOKEN_COUNTS <<< "${VLM_PROMPT_TOKEN_COUNTS-1 2 8 16}"
if (( ${#TOKEN_COUNTS[@]} == 0 )); then
  echo "VLM_PROMPT_TOKEN_COUNTS must contain positive integers" >&2
  exit 2
fi
declare -A SEEN_COUNTS=()
for count in "${TOKEN_COUNTS[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid VLM prompt token count: $count" >&2
    exit 2
  fi
  if [[ -n "${SEEN_COUNTS[$count]:-}" ]]; then
    echo "Duplicate VLM prompt token count: $count" >&2
    exit 2
  fi
  SEEN_COUNTS["$count"]=1
done
export CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/vlm-token-sweep}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/prompt-learning/vlm-token-sweep}"
RUN_PREFIX="${RUN_PREFIX:-pi05-vlm_only}"
run_one() {
  local count="$1" dry_run="$2"
  VLM_PROMPT_TOKENS="$count" RUN_NAME="${RUN_PREFIX}-vlm${count}-seed${SEED}" DRY_RUN="$dry_run" \
    bash "$SCRIPT_DIR/run_eval_libero-full.sh" vlm_only "$MODE" "$SEED"
}
echo "Checking all checkpoints: tokens=${TOKEN_COUNTS[*]} training_seed=$SEED"
for count in "${TOKEN_COUNTS[@]}"; do
  run_one "$count" true >/dev/null
done
for count in "${TOKEN_COUNTS[@]}"; do
  echo "Evaluating vlm_only: $count VLM prompt tokens"
  run_one "$count" "$DRY_RUN"
done
