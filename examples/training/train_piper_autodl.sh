#!/usr/bin/env bash
# AutoDL preset for four independent Piper policies. Run in the training conda env.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_autodl.sh [TASK] [VARIANT] [SEED] [extra train args...]

TASK: all (default), 1 (apple), 2 (cuboid), 3 (tennis), 4 (red cube)
VARIANT: full_reference (default), or an existing prompt-ablation variant
SEED: 0 (default)

Default: GPU 0; batch 8; FP32; 750 priming + 250 bridge + 12000 flow
updates; 16+16 prompts; CABO ratio 2; checkpoints at flow 6000, 9000 and 12000.
Temporary pretraining checkpoints are removed after each successful task.
Training predicts 50 actions; deployment executes 8 actions per observation.
All four tasks train separately, in sequence, from the same base checkpoint.

Set DATASET_BASE, PRETRAINED_PATH and TOKENIZER_PATH to the actual local paths.
Set RUN_GROUP to reuse a chosen output group, or leave it unset for a timestamp.
DRY_RUN=true validates metadata/paths and prints commands without using a GPU.
W&B is enabled by default (project piper-action-fit); WANDB_MODE=offline records locally.
WANDB_ENABLE=false disables W&B; local diagnostics still work. Model artifacts are not uploaded.
ACTION_EVAL_FREQ=500 evaluates 16 fixed observations per split at start, intervals and stage end.
ACTION_EVAL_SAMPLES=16, ACTION_EVAL_SEED=0, ACTION_UPDATE_FREQ=50 tune diagnostic cost.
EVAL_SPLIT=0.1 optionally reserves episodes for validation; default 0 keeps all training episodes.
GPU_IDS=0,1 selects two GPUs; NUM_PROCESSES defaults to the GPU list length.
FLOW_STEPS=3000 selects a shorter formal flow stage; SAVE_STEPS='[3000]' saves its final model.
SAVE_STEPS overrides SAVE_FREQ; KEEP_PRETRAIN_CHECKPOINT=true retains Stage-1 weights.
See examples/piper/README_TRAIN_AUTODL.md for setup, logs and checkpoint transfer.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-all}"
VARIANT="${2:-full_reference}"
SEED="${3:-0}"
if (( $# >= 3 )); then
  shift 3
else
  shift "$#"
fi

# Edit these three defaults, or export the variables before running this script.
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
export DATASET_BASE="${DATASET_BASE:-${WORK_ROOT}/datasets/May-pick-and-place}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-${WORK_ROOT}/models/pi05_libero_base}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${WORK_ROOT}/models/google/paligemma-3b-pt-224}"

export RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "${RUN_GROUP}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "RUN_GROUP must be a single label using letters, numbers, dots, dashes or underscores" >&2
  exit 2
fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/chkpt/2601-lerobot/piper/${RUN_GROUP}}"
export LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/piper/${RUN_GROUP}}"

export GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a PIPER_GPU_LIST <<< "${GPU_IDS}"
export NUM_PROCESSES="${NUM_PROCESSES:-${#PIPER_GPU_LIST[@]}}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export FLOW_STEPS="${FLOW_STEPS:-12000}"
export SAVE_FREQ="${SAVE_FREQ:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export KEEP_PRETRAIN_CHECKPOINT="${KEEP_PRETRAIN_CHECKPOINT:-false}"
export DTYPE="${DTYPE:-float32}"
case "${DTYPE}" in
  float32) EXPECTED_PRECISION=no ;;
  bfloat16) EXPECTED_PRECISION=bf16 ;;
  *) echo "DTYPE must be float32 or bfloat16" >&2; exit 2 ;;
esac
export MIXED_PRECISION="${MIXED_PRECISION:-${EXPECTED_PRECISION}}"
if [[ "${MIXED_PRECISION}" != "${EXPECTED_PRECISION}" ]]; then
  echo "DTYPE=${DTYPE} requires MIXED_PRECISION=${EXPECTED_PRECISION}" >&2
  exit 2
fi
export CABO_RATIO="${CABO_RATIO:-2.0}"
export CHUNK_SIZE="${CHUNK_SIZE:-50}"
# n_action_steps affects deployment, not the length of the flow training target.
export N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
export COMPILE_MODEL="${COMPILE_MODEL:-true}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export NTK_SAVE_STAGE_SNAPSHOTS="${NTK_SAVE_STAGE_SNAPSHOTS:-false}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
export NORMALIZATION_MODE="${NORMALIZATION_MODE:-QUANTILES}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "AutoDL Piper: task=${TASK} variant=${VARIANT} seed=${SEED} run_group=${RUN_GROUP}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log root: ${LOG_ROOT}"
exec bash "${SCRIPT_DIR}/train_pi05_real.sh" "${TASK}" "${VARIANT}" "${SEED}" \
  "--wandb.enable=${WANDB_ENABLE:-true}" \
  "--wandb.project=${WANDB_PROJECT:-piper-action-fit}" \
  "--wandb.mode=${WANDB_MODE:-online}" --wandb.disable_artifact=true \
  "--piper_eval.freq=${ACTION_EVAL_FREQ:-500}" \
  "--piper_eval.samples=${ACTION_EVAL_SAMPLES:-16}" \
  "--piper_eval.seed=${ACTION_EVAL_SEED:-0}" \
  "--piper_eval.execution_steps=${N_ACTION_STEPS}" \
  "--piper_eval.update_freq=${ACTION_UPDATE_FREQ:-50}" \
  "--dataset.eval_split=${EVAL_SPLIT:-0}" \
  "--num_workers=${NUM_WORKERS:-4}" "$@"
