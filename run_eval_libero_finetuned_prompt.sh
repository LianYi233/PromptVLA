#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_finetuned_prompt.sh [baseline|initial|trained|all] [SUITES] [TRAINING_SEED]
Defaults: all three conditions, all four LIBERO suites, training seed 0.
baseline: official v044, zero prompts; initial: the exact training step-0 dual prompts;
trained: CHECKPOINT_STEP (default 3000). Same EVAL_SEED (1000), episodes/task (10),
action execution steps (10), source dtype and saved normalization for all conditions.
All selected checkpoints are checked before any evaluation starts.

Overrides: PRETRAINED_PATH, TOKENIZER_PATH, CKPT_ROOT, RUN_NAME, BASE_OUTPUT,
VLM_PROMPT_TOKENS, ACTION_PROMPT_TOKENS, CHECKPOINT_STEP, EPISODES_PER_TASK,
EVAL_SEED, GPU_ID, DRY_RUN. Token counts must match the saved models.
Example: CHECKPOINT_STEP=1000 bash run_eval_libero_finetuned_prompt.sh trained all 0
EOF
  exit 0
fi
if (( $# > 3 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
CONDITION="${1:-all}"; SUITES="${2:-all}"; SEED="${3:-0}"
case "$CONDITION" in
  all) CONDITIONS=(baseline initial trained) ;;
  baseline|initial|trained) CONDITIONS=("$CONDITION") ;;
  *) echo "Unknown condition: $CONDITION" >&2; exit 2 ;;
esac
VLM_TOKENS="${VLM_PROMPT_TOKENS:-16}"; ACTION_TOKENS="${ACTION_PROMPT_TOKENS:-16}"
for count in "$VLM_TOKENS" "$ACTION_TOKENS" "${CHECKPOINT_STEP:-3000}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then echo "Token counts and trained step must be positive integers" >&2; exit 2; fi
done
export PRETRAINED_PATH="${PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_finetuned_v044}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
export CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/finetuned-v044}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/prompt-learning/finetuned-v044}"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export EVAL_SEED="${EVAL_SEED:-1000}"
export LEROBOT_EVAL_FINETUNED_PROMPT=1
if [[ -n "${BASE_CKPT:-}" || -n "${BASE_MODEL_PATH:-}" ]]; then
  echo "Use PRETRAINED_PATH, CKPT_ROOT and RUN_NAME; unset BASE_CKPT/BASE_MODEL_PATH" >&2; exit 2
fi
RUN_NAME="${RUN_NAME:-pi05-ftv044-dual-vlm${VLM_TOKENS}-act${ACTION_TOKENS}-seed${SEED}}"
TRAINED_STEP="${CHECKPOINT_STEP:-3000}"
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "DRY_RUN must be true or false" >&2; exit 2; fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then echo "Tokenizer not found: $TOKENIZER_PATH" >&2; exit 1; fi
python -m lerobot.scripts.libero_finetuned_prompt "$PRETRAINED_PATH"
run_one() {
  local condition="$1" dry="$2" step="$TRAINED_STEP"
  if [[ "$condition" == baseline ]]; then
    env -u VLM_PROMPT_TOKENS BASE_MODEL_PATH="$PRETRAINED_PATH" \
      RUN_NAME=pi05_libero_finetuned_v044-no_prompt DRY_RUN="$dry" \
      bash "$SCRIPT_DIR/run_eval_libero-full.sh" pi05_libero_base "$SUITES" "$SEED"
  else
    if [[ "$condition" == initial ]]; then step=0; fi
    env -u VLM_PROMPT_TOKENS RUN_NAME="$RUN_NAME" CHECKPOINT_STEP="$step" DRY_RUN="$dry" \
      bash "$SCRIPT_DIR/run_eval_libero-full.sh" dual_prompt_only "$SUITES" "$SEED"
  fi
}
for condition in "${CONDITIONS[@]}"; do
  if [[ "$condition" != baseline ]]; then
    step="$TRAINED_STEP"
    if [[ "$condition" == initial ]]; then step=0; fi
    checkpoint="$CKPT_ROOT/$RUN_NAME/checkpoints/$(printf '%06d' "$step")/pretrained_model"
    python -m lerobot.scripts.libero_finetuned_prompt "$PRETRAINED_PATH" --checkpoint "$checkpoint" \
      --vlm-tokens "$VLM_TOKENS" --action-tokens "$ACTION_TOKENS"
  fi
  run_one "$condition" true >/dev/null
done
for condition in "${CONDITIONS[@]}"; do
  echo "Evaluating finetuned-v044 condition=$condition"
  run_one "$condition" "$DRY_RUN"
done
