#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_vlm_only.sh [base|finetuned|both] [baseline|initial|trained|all] [SUITES] [SEED]
Defaults: both all all 0. baseline/initial/all require the exact saved step-0 checkpoint.
trained also supports SAVE_ONLY_FINAL runs without step-0, validating against their source instead.
baseline: original source weights + step-0 config/processors + zero prompts.
initial: actual random VLM prompt before any update; trained: CHECKPOINT_STEP (3000).
Default: 10 episodes/task, EVAL_SEED=1000, n_action_steps=10, all four suites.
Both sources are evaluated sequentially on GPU_ID=0; use separate invocations/GPU_IDs to parallelize.

Overrides: BASE_PRETRAINED_PATH, FINETUNED_PRETRAINED_PATH, TOKENIZER_PATH, CKPT_ROOT,
BASE_OUTPUT, VLM_PROMPT_TOKENS (1), CHECKPOINT_STEP, EPISODES_PER_TASK, EVAL_SEED, GPU_ID, DRY_RUN.
PRETRAINED_PATH and RUN_NAME may be used only when selecting a single source.
DRY_RUN validates all selected conditions and prints commands without creating baseline files.
EOF
  exit 0
fi
if (( $# > 4 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
SELECT="${1:-both}"; CONDITION="${2:-all}"; SUITES="${3:-all}"; SEED="${4:-0}"
case "$SELECT" in both) SOURCES=(base finetuned) ;; base|finetuned) SOURCES=("$SELECT") ;; *) echo "Invalid source" >&2; exit 2 ;; esac
case "$CONDITION" in all) CONDITIONS=(baseline initial trained) ;; baseline|initial|trained) CONDITIONS=("$CONDITION") ;; *) echo "Invalid condition" >&2; exit 2 ;; esac
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo "Invalid suite" >&2; exit 2 ;; esac
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid SEED" >&2; exit 2; fi
if [[ -n "${BASE_CKPT:-}" || -n "${BASE_MODEL_PATH:-}" ]]; then
  echo "Unset BASE_CKPT/BASE_MODEL_PATH; use documented source/root variables" >&2; exit 2
fi
if [[ "$SELECT" == both && ( -n "${RUN_NAME:-}" || -n "${PRETRAINED_PATH:-}" ) ]]; then
  echo "RUN_NAME/PRETRAINED_PATH require a single source" >&2; exit 2
fi
TOKENS="${VLM_PROMPT_TOKENS:-1}"; STEP="${CHECKPOINT_STEP:-3000}"
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
for value in "$TOKENS" "$STEP" "$EPISODES_PER_TASK"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Tokens/step/episodes must be positive integers" >&2; exit 2; fi
done
if [[ ! "$EVAL_SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid EVAL_SEED" >&2; exit 2; fi
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
export CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/vlm-only}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/vlm-only}"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LEROBOT_EVAL_FINETUNED_PROMPT=1
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "Invalid DRY_RUN" >&2; exit 2; fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then echo "Tokenizer directory not found: $TOKENIZER_PATH" >&2; exit 1; fi
select_source() {
  KIND="$1"
  if [[ "$KIND" == base ]]; then
    SOURCE_PATH="${PRETRAINED_PATH:-${BASE_PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_base}}"
  else
    SOURCE_PATH="${PRETRAINED_PATH:-${FINETUNED_PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_finetuned_v044}}"
  fi
  NAME="${RUN_NAME:-pi05-${KIND}-vlm${TOKENS}-seed${SEED}}"
  INITIAL="$CKPT_ROOT/$NAME/checkpoints/000000/pretrained_model"
  TRAINED="$CKPT_ROOT/$NAME/checkpoints/$(printf '%06d' "$STEP")/pretrained_model"
  BASELINE="$CKPT_ROOT/$NAME/matched_baseline/pretrained_model"
}
# Validate every selected source and checkpoint before creating baselines or running GPU jobs.
for kind in "${SOURCES[@]}"; do
  select_source "$kind"
  CHECK_ARGS=("$SOURCE_PATH" --kind "$KIND" --tokens "$TOKENS")
  if [[ -e "$INITIAL" || -L "$INITIAL" ]]; then
    CHECK_ARGS+=(--initial "$INITIAL")
  elif [[ "$CONDITION" != trained ]]; then
    echo "Condition '$CONDITION' requires the saved step-0 checkpoint: $INITIAL" >&2
    echo "For SAVE_ONLY_FINAL runs, select 'trained' to evaluate the final checkpoint." >&2
    exit 1
  else
    echo "No step-0 checkpoint; validating trained prompt, processors and source provenance directly."
  fi
  if [[ "$CONDITION" == all || "$CONDITION" == trained ]]; then CHECK_ARGS+=(--checkpoint "$TRAINED"); fi
  if [[ "$CONDITION" == all || "$CONDITION" == baseline ]]; then CHECK_ARGS+=(--baseline-output "$BASELINE" --dry-run); fi
  python -m lerobot.scripts.libero_vlm_only "${CHECK_ARGS[@]}"
done
for kind in "${SOURCES[@]}"; do
  select_source "$kind"
  for condition in "${CONDITIONS[@]}"; do
    echo "Evaluating source=$kind condition=$condition VLM=$TOKENS action=0"
    if [[ "$condition" == baseline ]]; then
      if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' env -u VLM_PROMPT_TOKENS "BASE_MODEL_PATH=$BASELINE" "RUN_NAME=$NAME-zero-prompt" \
          "BASE_OUTPUT=$BASE_OUTPUT" "EVAL_SEED=$EVAL_SEED" "DRY_RUN=false" \
          bash "$SCRIPT_DIR/run_eval_libero-full.sh" pi05_libero_base "$SUITES" "$SEED"
        printf '\n'
        continue
      fi
      python -m lerobot.scripts.libero_vlm_only "$SOURCE_PATH" --kind "$KIND" --tokens "$TOKENS" \
        --initial "$INITIAL" --baseline-output "$BASELINE"
      env -u VLM_PROMPT_TOKENS BASE_MODEL_PATH="$BASELINE" RUN_NAME="$NAME-zero-prompt" \
        bash "$SCRIPT_DIR/run_eval_libero-full.sh" pi05_libero_base "$SUITES" "$SEED"
    else
      EVAL_STEP="$STEP"; if [[ "$condition" == initial ]]; then EVAL_STEP=0; fi
      env -u VLM_PROMPT_TOKENS RUN_NAME="$NAME" CHECKPOINT_STEP="$EVAL_STEP" DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/run_eval_libero-full.sh" vlm_only "$SUITES" "$SEED"
    fi
  done
done
