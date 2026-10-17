#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_finetuned_prompt.sh [SEED] [extra args...]
Continue lerobot/pi05_libero_finetuned_v044 with ONLY VLM and action prompts.
Defaults: 16+16 tokens, 3000 flow steps, batch 32, seed 0, CABO/priming off.
Load the source config, dtype and saved normalization; do not replace dataset stats.
Save the exact step-0 prompts and checkpoints every 1000 steps (plus the final step).
W&B is enabled: project=prompt-learning, mode=online, disable_artifact=true.

Overrides: PRETRAINED_PATH, DATASET_ROOT, DATASET_REPO_ID, TOKENIZER_PATH,
VLM_PROMPT_TOKENS, ACTION_PROMPT_TOKENS, FLOW_STEPS, SAVE_FREQ, GPU_IDS,
NUM_PROCESSES, BATCH_SIZE, OUTPUT_ROOT, LOG_ROOT, RUN_NAME, DRY_RUN.
Extra args are limited to --wandb.*, --policy.optimizer_*, --policy.scheduler_*,
--num_workers, --log_freq, --policy.compile_model and --policy.gradient_checkpointing.
Resume an interrupted run with lerobot-train --config_path=.../train_config.json --resume=true.
Evaluate with: bash run_eval_libero_finetuned_prompt.sh all all 0
EOF
  exit 0
fi
SEED="${1:-0}"
if (( $# )); then shift; fi
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid SEED: $SEED" >&2; exit 2; fi
EXTRA_ARGS=("$@")
for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --wandb.*=*|--policy.optimizer_*=*|--policy.scheduler_*=*|--num_workers=*|--log_freq=*|\
    --policy.compile_model=*|--policy.gradient_checkpointing=*) ;;
    *) echo "Unsupported override: $arg; use --help for experiment settings" >&2; exit 2 ;;
  esac
done
VLM_PROMPT_TOKENS="${VLM_PROMPT_TOKENS:-16}"
ACTION_PROMPT_TOKENS="${ACTION_PROMPT_TOKENS:-16}"
FLOW_STEPS="${FLOW_STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
for key in VLM_PROMPT_TOKENS ACTION_PROMPT_TOKENS FLOW_STEPS SAVE_FREQ BATCH_SIZE NUM_PROCESSES; do
  if [[ ! "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "$key must be a positive integer" >&2; exit 2; fi
done
PRETRAINED_PATH="${PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_finetuned_v044}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets/libero}"
DATASET_REPO_ID="${DATASET_REPO_ID:-libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/finetuned-v044}"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/prompt-learning/finetuned-v044}"
RUN_NAME="${RUN_NAME:-pi05-ftv044-dual-vlm${VLM_PROMPT_TOKENS}-act${ACTION_PROMPT_TOKENS}-seed${SEED}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "DRY_RUN must be true or false" >&2; exit 2; fi
for directory in "$PRETRAINED_PATH" "$DATASET_ROOT" "$TOKENIZER_PATH"; do
  if [[ ! -d "$directory" ]]; then echo "Directory not found: $directory" >&2; exit 1; fi
done
if [[ -e "$OUTPUT_DIR" ]]; then echo "Output already exists; refusing to overwrite: $OUTPUT_DIR" >&2; exit 1; fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m lerobot.scripts.libero_finetuned_prompt "$PRETRAINED_PATH"
TRAIN_ARGS=(
  --policy.path="$PRETRAINED_PATH"
  --policy.tokenizer_name="$TOKENIZER_PATH"
  --policy.device=cuda --policy.use_amp=false
  --policy.compile_model=true --policy.gradient_checkpointing=true
  --policy.num_vlm_prompt_tokens="$VLM_PROMPT_TOKENS"
  --policy.num_prompt_tokens="$ACTION_PROMPT_TOKENS"
  --policy.training_stage=flow
  --policy.next_action_pretrain_steps=0 --policy.next_action_bridge_steps=0
  --policy.cabo_enabled=false --policy.train_action_projections=false --policy.use_peft=false
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true
  --policy.ntk_save_stage_snapshots=true --policy.push_to_hub=false
  --preserve_pretrained_normalization=true
  --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATASET_ROOT" --dataset.video_backend=pyav
  --output_dir="$OUTPUT_DIR" --job_name="$RUN_NAME"
  --steps="$FLOW_STEPS" --batch_size="$BATCH_SIZE" --seed="$SEED"
  --env_eval_freq=0 --save_checkpoint=true --save_freq="$SAVE_FREQ"
  --wandb.enable=true --wandb.project=prompt-learning --wandb.mode=online --wandb.disable_artifact=true
)
LAUNCH_ARGS=(launch --num_processes="$NUM_PROCESSES" --mixed_precision=bf16)
if (( NUM_PROCESSES > 1 )); then LAUNCH_ARGS+=(--multi_gpu); fi
echo "source=$PRETRAINED_PATH prompts=$VLM_PROMPT_TOKENS+$ACTION_PROMPT_TOKENS steps=$FLOW_STEPS"
echo "output=$OUTPUT_DIR (step 0 and every $SAVE_FREQ updates)"
if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_IDS:-0}" accelerate "${LAUNCH_ARGS[@]}" \
    lerobot-train "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'
  exit 0
fi
command -v accelerate >/dev/null
TRAINER=$(command -v lerobot-train)
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" accelerate "${LAUNCH_ARGS[@]}" "$TRAINER" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/$RUN_NAME.log"
