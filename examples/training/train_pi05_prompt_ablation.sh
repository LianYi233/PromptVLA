#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash train_pi05_prompt_ablation.sh VARIANT [SEED] [extra lerobot-train args...]

Variants:
  full_reference    750 inpainting + 250 bridge + 3000 flow, dual prompts, CABO
  no_bridge         1000 inpainting + 0 bridge + 3000 flow, dual prompts, CABO
  direct_dual       0 pretraining + 3000 flow, dual prompts, CABO
  dual_prompt_only  0 pretraining + 3000 flow, dual prompts, CABO off
  vlm_only          0 pretraining + 3000 flow, VLM prompt only, CABO off
  action_only       0 pretraining + 3000 flow, action prompt only, CABO off
  no_cabo           750 inpainting + 250 bridge + 3000 flow, dual prompts, CABO off

All variants use 3000 flow updates by default. Set FLOW_STEPS to override this
for a one-off run.
full_reference automatically saves exact initialization and priming-end snapshots,
and saves flow checkpoints every 1000 updates (including flow 2000 = total 3000).
Set NTK_SAVE_STAGE_SNAPSHOTS=false to disable extra stage snapshots, or SAVE_FREQ
to override the flow checkpoint interval. Other variants keep their existing defaults.
Set RUN_NAME to isolate tasks, or DRY_RUN=true to print the command without training.
Set VLM_PROMPT_TOKENS to a positive integer to change the active VLM prompt bank:
  VLM_PROMPT_TOKENS=8 bash examples/training/train_pi05_prompt_ablation.sh vlm_only 0
An explicit token count adds -vlmN to the default run name (even for N=16).
Without this override, the original 16-token recipe and run name are preserved.
For sequential runs, use train_pi05_vlm_token_sweep.sh [SEED] (defaults: 4 then 32).
Set VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" on that sweep script for all six lengths.
EOF
}

VARIANT="${1:-}"
if [[ "${VARIANT}" == --help || "${VARIANT}" == -h ]]; then
  usage
  exit 0
fi
if [[ -z "${VARIANT}" ]]; then
  usage
  exit 2
fi
SEED="${2:-0}"
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi
EXTRA_ARGS=("$@")

VLM_PROMPT_TOKENS_OVERRIDE="${VLM_PROMPT_TOKENS:-}"
VLM_PROMPT_TOKENS="${VLM_PROMPT_TOKENS:-16}"
if [[ ! "${VLM_PROMPT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLM_PROMPT_TOKENS must be a positive integer without leading zeros" >&2
  exit 2
fi
ACTION_PROMPT_TOKENS=16
CABO_ENABLED=true
PRETRAIN_STEPS=1000
BRIDGE_STEPS=250
FLOW_STEPS_OVERRIDE="${FLOW_STEPS:-}"
FLOW_STEPS_DEFAULT=3000
SAVE_FREQ_DEFAULT=3000
NTK_SNAPSHOTS_DEFAULT=false

case "${VARIANT}" in
  full_reference)
    SAVE_FREQ_DEFAULT=1000
    NTK_SNAPSHOTS_DEFAULT=true
    ;;
  no_bridge)
    BRIDGE_STEPS=0
    ;;
  direct_dual)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    ;;
  dual_prompt_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    CABO_ENABLED=false
    ;;
  vlm_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    ACTION_PROMPT_TOKENS=0
    CABO_ENABLED=false
    ;;
  action_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    VLM_PROMPT_TOKENS=0
    CABO_ENABLED=false
    ;;
  no_cabo)
    CABO_ENABLED=false
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    usage >&2
    exit 2
    ;;
esac

FLOW_STEPS="${FLOW_STEPS_OVERRIDE:-${FLOW_STEPS_DEFAULT}}"

DATASET_REPO_ID="${DATASET_REPO_ID:-libero}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets/libero}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_base}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation}"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/prompt-ablation}"
GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-${SAVE_FREQ_DEFAULT}}"
NTK_SAVE_STAGE_SNAPSHOTS="${NTK_SAVE_STAGE_SNAPSHOTS:-${NTK_SNAPSHOTS_DEFAULT}}"
CABO_RATIO="${CABO_RATIO:-2.0}"
MASKED_STEPS="${MASKED_STEPS:-40}"
DTYPE="${DTYPE:-float32}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

DEFAULT_RUN_NAME="pi05-${VARIANT}-seed${SEED}"
if [[ -n "${VLM_PROMPT_TOKENS_OVERRIDE}" && "${VARIANT}" != action_only ]]; then
  DEFAULT_RUN_NAME="pi05-${VARIANT}-vlm${VLM_PROMPT_TOKENS}-seed${SEED}"
fi
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
DRY_RUN="${DRY_RUN:-false}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "Pretrained model does not exist: ${PRETRAINED_PATH}" >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer does not exist: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
  exit 1
fi
if (( PRETRAIN_STEPS > 0 )) && [[ -e "${OUTPUT_DIR}_next_action_pretrain" ]]; then
  echo "Pretraining output already exists; choose a new RUN_NAME: ${OUTPUT_DIR}_next_action_pretrain" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != true ]] && ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not available in the active environment" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != true ]] && ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train is not available; install this checkout with: pip install -e ." >&2
  exit 1
fi

TRAIN_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --policy.type=pi05
  --policy.training_stage=flow
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.tokenizer_name="${TOKENIZER_PATH}"
  --policy.device=cuda
  --policy.dtype="${DTYPE}"
  --policy.compile_model="${COMPILE_MODEL}"
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
  --policy.num_vlm_prompt_tokens="${VLM_PROMPT_TOKENS}"
  --policy.num_prompt_tokens="${ACTION_PROMPT_TOKENS}"
  --policy.next_action_pretrain_steps="${PRETRAIN_STEPS}"
  --policy.next_action_bridge_steps="${BRIDGE_STEPS}"
  --policy.next_action_masked_steps="${MASKED_STEPS}"
  --policy.ntk_save_stage_snapshots="${NTK_SAVE_STAGE_SNAPSHOTS}"
  --policy.cabo_enabled="${CABO_ENABLED}"
  --policy.cabo_prompt_update_ratio="${CABO_RATIO}"
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}"
  --steps="${FLOW_STEPS}"
  --batch_size="${BATCH_SIZE}"
  --seed="${SEED}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
)
if [[ -n "${SAVE_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--save_steps="${SAVE_STEPS}")
fi

LAUNCH_ARGS=(
  launch
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi

echo "variant=${VARIANT} seed=${SEED} GPUs=${GPU_IDS} processes=${NUM_PROCESSES}"
echo "prompts=${VLM_PROMPT_TOKENS}+${ACTION_PROMPT_TOKENS} pretrain=${PRETRAIN_STEPS} bridge=${BRIDGE_STEPS} flow=${FLOW_STEPS} CABO=${CABO_ENABLED}"
echo "output=${OUTPUT_DIR}"
if [[ -n "${SAVE_STEPS:-}" ]]; then
  echo "NTK stage snapshots=${NTK_SAVE_STAGE_SNAPSHOTS} flow checkpoints=${SAVE_STEPS} (plus final step; SAVE_FREQ ignored)"
else
  echo "NTK stage snapshots=${NTK_SAVE_STAGE_SNAPSHOTS} flow checkpoint interval=${SAVE_FREQ}"
fi
if [[ "${NTK_SAVE_STAGE_SNAPSHOTS}" == true ]] && (( PRETRAIN_STEPS > 0 )); then
  PRETRAIN_OUTPUT_DIR="${OUTPUT_DIR}_next_action_pretrain"
  printf 'NTK before training: %s/checkpoints/%06d/pretrained_model\n' "${PRETRAIN_OUTPUT_DIR}" 0
  if (( BRIDGE_STEPS > 0 )); then
    printf 'NTK after priming:  %s/checkpoints/%06d/pretrained_model\n' \
      "${PRETRAIN_OUTPUT_DIR}" "$((PRETRAIN_STEPS - BRIDGE_STEPS))"
  fi
  printf 'NTK after stage 2:  %s/checkpoints/%06d/pretrained_model\n' "${PRETRAIN_OUTPUT_DIR}" "${PRETRAIN_STEPS}"
  printf 'NTK final flow:     %s/checkpoints/%06d/pretrained_model (cumulative updates: %d)\n' \
    "${OUTPUT_DIR}" "${FLOW_STEPS}" "$((PRETRAIN_STEPS + FLOW_STEPS))"
fi

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_IDS}" accelerate "${LAUNCH_ARGS[@]}" \
    lerobot-train "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  accelerate "${LAUNCH_ARGS[@]}" "$(command -v lerobot-train)" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
