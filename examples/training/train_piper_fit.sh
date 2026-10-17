#!/usr/bin/env bash
# Controlled fitting diagnostics before another four-task training run.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_fit.sh TASK [PROFILE] [SEED] [extra train args...]

TASK: 1,2,3,4 or all. PROFILE (default projections):
  reference     Original prompt-only LR 2.5e-5, CABO on
  prompt_lr     Prompt-only LR 1e-4, CABO on
  no_cabo       Same LR 1e-4, CABO off
  projections   Same as no_cabo, also train action_in_proj and action_out_proj

All profiles use 750 priming + 250 bridge, 50 predicted / 8 executed actions,
FP32 and the same data/normalization. Change one factor at each comparison.
projections has 115,744 trainable parameters for the default model; it is not
the 49,152-parameter prompt-only method. Neither backbone is unfrozen.

Defaults: fit episode [0] for 2000 formal flow steps, save only final checkpoint.
FIT_EPISODES='[0,1,2]' selects a few episodes; FIT_EPISODES=all selects all data.
FLOW_STEPS, SAVE_STEPS, BATCH_SIZE, WORK_ROOT, DATASET_BASE, PRETRAINED_PATH,
TOKENIZER_PATH, OUTPUT_ROOT, LOG_ROOT and GPU_IDS work as in train_piper_autodl.sh.
DRY_RUN=true checks paths/metadata and prints commands, without using a GPU.
RUN_GROUP defaults to a unique timestamp plus profile; keep separate output groups.
Do not pass --resume here. Use the base checkpoint for matched comparisons.
Evaluate the same training episode(s) with eval_piper_offline.py after fitting.
W&B and periodic recorded-action metrics are enabled as in train_piper_autodl.sh.
Keep EVAL_SPLIT=0 for a one-episode fit; use a split only with multiple episodes.
EOF
  exit 0
fi

TASK="${1:?Specify task 1,2,3,4 or all; use --help}"
PROFILE="${2:-projections}"
SEED="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi

TRAIN_PROJECTIONS=false
VARIANT=full_reference
PROMPT_LR=0.0001
case "${PROFILE}" in
  reference) PROMPT_LR=0.000025 ;;
  prompt_lr) ;;
  no_cabo) VARIANT=no_cabo ;;
  projections) VARIANT=no_cabo; TRAIN_PROJECTIONS=true ;;
  *) echo "Unknown profile ${PROFILE}; use --help" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_GROUP="${RUN_GROUP:-piper-fit-${PROFILE}-$(date +%Y%m%d-%H%M%S)}"
export FLOW_STEPS="${FLOW_STEPS:-2000}"
export SAVE_STEPS="${SAVE_STEPS:-[${FLOW_STEPS}]}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
# Avoid lengthy recompilation for the first fitting diagnosis; can be explicitly enabled.
export COMPILE_MODEL="${COMPILE_MODEL:-false}"
FIT_EPISODES="${FIT_EPISODES:-[0]}"
FIT_ARGS=(
  "--policy.train_action_projections=${TRAIN_PROJECTIONS}"
  "--policy.optimizer_lr=${PROMPT_LR}"
  --policy.scheduler_decay_lr=0.00001
  --dataset.image_transforms.enable=false
  --log_freq=50
)
if [[ "${FIT_EPISODES}" != all ]]; then
  FIT_ARGS+=("--dataset.episodes=${FIT_EPISODES}")
fi
echo "Piper fit: profile=${PROFILE}, episodes=${FIT_EPISODES}, LR=${PROMPT_LR}, projections=${TRAIN_PROJECTIONS}"
exec bash "${SCRIPT_DIR}/train_piper_autodl.sh" "${TASK}" "${VARIANT}" "${SEED}" "${FIT_ARGS[@]}" "$@"
