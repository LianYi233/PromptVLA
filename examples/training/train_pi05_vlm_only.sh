#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_only.sh base|finetuned [SEED] [extra args...]
PI05 + one VLM prompt bank; default 1 token, action prompts=0, all other weights frozen.
Ordinary flow only: no CABO, priming, bridge, LoRA or action projection training.
TRAIN_ACTION_EXPERT=true explicitly selects VLM prompt + full action network instead;
it requires a distinct RUN_NAME and cannot export a prompt-only initialization.
Defaults: 3000 steps, batch 32 PER GPU, bf16, W&B online/project prompt-learning.
SAVE_ONLY_FINAL=true saves only the final checkpoint (003000 by default), including no step-0 snapshot.
base uses dataset QUANTILES (as in the earlier ablation); finetuned keeps v044's saved MEAN_STD.
The evaluator builds a zero-prompt baseline using each run's exact initial processors.

GPU_IDS=0,1 BATCH_SIZE=16 bash examples/training/train_pi05_vlm_only.sh base 0
For two independent runs: bash examples/training/train_pi05_vlm_only_pair.sh 0

Environment: VLM_PROMPT_TOKENS, GPU_IDS (process count inferred), NUM_PROCESSES,
MAIN_PROCESS_PORT, BATCH_SIZE, FLOW_STEPS, SAVE_FREQ, SAVE_ONLY_FINAL, DTYPE (bfloat16|float32),
PROMPT_INIT_PATH optionally imports an exported step-zero .safetensors prompt;
SEED controls other training randomness unless PDET is enabled. Requires --policy.compile_model=false.
PROMPT_EXPORT_PATH instead exports the normally constructed initial prompt and exits with zero
updates (no W&B, optimizer, probes or full checkpoint). Incompatible with PROMPT_INIT_PATH.
PROMPT_EXPORT_REFERENCE optionally verifies the replaced-prompt hash in an earlier prompt_init.json.
PRETRAINED_PATH, BASE_PRETRAINED_PATH, FINETUNED_PRETRAINED_PATH, DATASET_ROOT,
DATASET_REPO_ID, TOKENIZER_PATH, OUTPUT_ROOT, LOG_ROOT, RUN_NAME, DRY_RUN.
Extra args (use --key=value): --wandb.*, --policy.optimizer_*, --policy.scheduler_*,
--policy.compile_model, --policy.gradient_checkpointing, --num_workers, --log_freq.
--prompt_rank.* enables fixed-probe velocity-to-prompt Jacobian spectra (requires no compile).
--pdet.* separates data order, flow noise and flow time; prefer train_pi05_vlm_only_pdet_sweep.sh.
EOF
  exit 0
fi
SOURCE="${1:-}"; SEED="${2:-0}"
case "$SOURCE" in base|finetuned) ;; *) echo "Source must be base or finetuned; see --help" >&2; exit 2 ;; esac
if (( $# >= 2 )); then shift 2; else shift; fi
EXTRA_ARGS=("$@")
for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --wandb.*=*|--policy.optimizer_*=*|--policy.scheduler_*=*|--policy.compile_model=*|\
    --policy.gradient_checkpointing=*|--num_workers=*|--log_freq=*|--prompt_rank.*=*|--pdet.*=*|--keyframe_loss.*=*) ;;
    *) echo "Unsupported override: $arg; use --help for VLM-only controls" >&2; exit 2 ;;
  esac
done
VLM_PROMPT_TOKENS="${VLM_PROMPT_TOKENS:-1}"
TRAIN_ACTION_EXPERT="${TRAIN_ACTION_EXPERT:-false}"
case "$TRAIN_ACTION_EXPERT" in
  false) VLM_ONLY=true ;;
  true)
    VLM_ONLY=false
    [[ -n "${RUN_NAME:-}" && -z "${PROMPT_EXPORT_PATH:-}" ]] || {
      echo 'Action-expert training requires explicit RUN_NAME and forbids prompt-only export' >&2; exit 2;
    } ;;
  *) echo 'TRAIN_ACTION_EXPERT must be true or false' >&2; exit 2 ;;
esac
FLOW_STEPS="${FLOW_STEPS:-3000}"; SAVE_FREQ="${SAVE_FREQ:-1000}"; BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_ONLY_FINAL="${SAVE_ONLY_FINAL:-false}"
case "$SAVE_ONLY_FINAL" in
  true) SAVE_FREQ="$FLOW_STEPS"; SAVE_INITIAL=false ;;
  false) SAVE_INITIAL=true ;;
  *) echo "SAVE_ONLY_FINAL must be true or false" >&2; exit 2 ;;
esac
GPU_IDS="${GPU_IDS:-0}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo "GPU_IDS must be comma-separated GPU indices" >&2; exit 2; fi
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then echo "Duplicate GPU: $gpu" >&2; exit 2; fi
  SEEN_GPUS[$gpu]=1
done
NUM_PROCESSES="${NUM_PROCESSES:-${#GPUS[@]}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29531}"
for key in VLM_PROMPT_TOKENS FLOW_STEPS SAVE_FREQ BATCH_SIZE NUM_PROCESSES MAIN_PROCESS_PORT; do
  if [[ ! "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "$key must be a positive integer" >&2; exit 2; fi
done
if (( NUM_PROCESSES != ${#GPUS[@]} || MAIN_PROCESS_PORT > 65535 )); then
  echo "NUM_PROCESSES must equal visible GPU count; port must be <= 65535" >&2; exit 2
fi
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid SEED" >&2; exit 2; fi
DTYPE="${DTYPE:-bfloat16}"
case "$DTYPE" in bfloat16) PRECISION=bf16 ;; float32) PRECISION=no ;; *) echo "Invalid DTYPE" >&2; exit 2 ;; esac
if [[ "$SOURCE" == base ]]; then
  PRETRAINED_PATH="${PRETRAINED_PATH:-${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}}"
  PRESERVE=false
  NORM='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}'
else
  PRETRAINED_PATH="${PRETRAINED_PATH:-${FINETUNED_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_finetuned_v044}}"
  PRESERVE=true
  NORM='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
fi
DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
RUN_NAME="${RUN_NAME:-pi05-${SOURCE}-vlm${VLM_PROMPT_TOKENS}-seed${SEED}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "DRY_RUN must be true or false" >&2; exit 2; fi
if [[ -n "${PROMPT_EXPORT_PATH:-}" && -n "${PROMPT_INIT_PATH:-}" ]]; then
  echo "PROMPT_EXPORT_PATH cannot be combined with PROMPT_INIT_PATH" >&2; exit 2
fi
if [[ -n "${PROMPT_EXPORT_REFERENCE:-}" && -z "${PROMPT_EXPORT_PATH:-}" ]]; then
  echo "PROMPT_EXPORT_REFERENCE requires PROMPT_EXPORT_PATH" >&2; exit 2
fi
for directory in "$PRETRAINED_PATH" "$DATASET_ROOT" "$TOKENIZER_PATH"; do
  if [[ ! -d "$directory" ]]; then echo "Directory not found: $directory" >&2; exit 1; fi
done
if [[ -e "$OUTPUT_DIR" ]]; then echo "Output already exists; choose another RUN_NAME: $OUTPUT_DIR" >&2; exit 1; fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m lerobot.scripts.libero_vlm_only "$PRETRAINED_PATH" --kind "$SOURCE" \
  --dataset-root "$DATASET_ROOT" --tokens "$VLM_PROMPT_TOKENS"
if [[ -n "${PROMPT_INIT_PATH:-}" ]]; then
  python -m lerobot.scripts.export_pi05_prompt_init check "$PROMPT_INIT_PATH" --tokens "$VLM_PROMPT_TOKENS"
fi
if [[ -n "${PROMPT_EXPORT_PATH:-}" ]]; then
  EXPORT_CHECK=(check-export "$PROMPT_EXPORT_PATH" --seed "$SEED" --tokens "$VLM_PROMPT_TOKENS")
  if [[ -n "${PROMPT_EXPORT_REFERENCE:-}" ]]; then EXPORT_CHECK+=(--reference "$PROMPT_EXPORT_REFERENCE"); fi
  python -m lerobot.scripts.export_pi05_prompt_init "${EXPORT_CHECK[@]}"
fi
TRAIN_ARGS=(
  --policy.path="$PRETRAINED_PATH" --policy.tokenizer_name="$TOKENIZER_PATH"
  --policy.device=cuda --policy.dtype="$DTYPE" --policy.use_amp=false
  --policy.compile_model=true --policy.gradient_checkpointing=true
  --policy.vlm_prompt_only="$VLM_ONLY" --policy.num_vlm_prompt_tokens="$VLM_PROMPT_TOKENS"
  --policy.num_prompt_tokens=0 --policy.training_stage=flow
  --policy.next_action_pretrain_steps=0 --policy.next_action_bridge_steps=0
  --policy.cabo_enabled=false --policy.train_action_projections=false --policy.use_peft=false
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true
  --policy.normalization_mapping="$NORM" --preserve_pretrained_normalization="$PRESERVE"
  --policy.ntk_save_stage_snapshots="$SAVE_INITIAL" --policy.push_to_hub=false
  --policy.optimizer_lr=2.5e-5 --policy.optimizer_grad_clip_norm=0.0
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=1e-5
  --dataset.repo_id="${DATASET_REPO_ID:-libero}" --dataset.root="$DATASET_ROOT" --dataset.video_backend=pyav
  --output_dir="$OUTPUT_DIR" --job_name="$RUN_NAME" --steps="$FLOW_STEPS" --batch_size="$BATCH_SIZE"
  --seed="$SEED" --env_eval_freq=0 --save_checkpoint=true --save_freq="$SAVE_FREQ"
  --wandb.enable=true --wandb.project=prompt-learning --wandb.mode=online --wandb.disable_artifact=true
)
if [[ -n "${PROMPT_INIT_PATH:-}" ]]; then TRAIN_ARGS+=(--vlm_prompt_init_path="$PROMPT_INIT_PATH"); fi
if [[ "$TRAIN_ACTION_EXPERT" == true ]]; then TRAIN_ARGS+=(--policy.train_action_expert=true); fi
if [[ -n "${PROMPT_EXPORT_PATH:-}" ]]; then TRAIN_ARGS+=(--vlm_prompt_export_path="$PROMPT_EXPORT_PATH"); fi
if [[ -n "${PROMPT_EXPORT_REFERENCE:-}" ]]; then TRAIN_ARGS+=(--vlm_prompt_export_reference="$PROMPT_EXPORT_REFERENCE"); fi
if [[ "$SAVE_ONLY_FINAL" == true ]]; then TRAIN_ARGS+=(--save_steps='[]'); fi
LAUNCH_ARGS=(launch --num_machines=1 --num_processes="$NUM_PROCESSES" --mixed_precision="$PRECISION"
  --main_process_port="$MAIN_PROCESS_PORT" --dynamo_backend=no)
if (( NUM_PROCESSES > 1 )); then LAUNCH_ARGS+=(--multi_gpu); fi
if [[ "$TRAIN_ACTION_EXPERT" == true ]]; then
  echo "source=$SOURCE VLM=$VLM_PROMPT_TOKENS action_prompts=0 trainable=VLM-prompt+action-transformer+action-maps+time-MLPs dtype=$DTYPE"
else
  echo "source=$SOURCE VLM=$VLM_PROMPT_TOKENS action=0 trainable=$((VLM_PROMPT_TOKENS * 2048)) dtype=$DTYPE"
fi
echo "GPU_IDS=$GPU_IDS per_device_batch=$BATCH_SIZE global_batch=$((BATCH_SIZE * NUM_PROCESSES)) port=$MAIN_PROCESS_PORT"
if [[ -n "${PROMPT_EXPORT_PATH:-}" ]]; then
  echo "initialization only: export=$PROMPT_EXPORT_PATH (0 updates; no optimizer/probes/full checkpoint/W&B)"
elif [[ "$SAVE_ONLY_FINAL" == true ]]; then
  echo "output=$OUTPUT_DIR (final only=$FLOW_STEPS; no initial/intermediate checkpoints)"
else
  echo "output=$OUTPUT_DIR (initial=000000, interval=$SAVE_FREQ, final=$FLOW_STEPS)"
fi
if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU_IDS" accelerate "${LAUNCH_ARGS[@]}" \
    lerobot-train "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'; exit 0
fi
command -v accelerate >/dev/null
TRAINER=$(command -v lerobot-train)
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export TMPDIR="${TMPDIR:-$OUTPUT_ROOT/.runtime/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_ROOT/.runtime/cache/torchinductor/$RUN_NAME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUTPUT_ROOT/.runtime/cache/triton/$RUN_NAME}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate "${LAUNCH_ARGS[@]}" "$TRAINER" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/$RUN_NAME.log"
