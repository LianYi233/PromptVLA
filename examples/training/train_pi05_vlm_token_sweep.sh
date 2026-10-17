#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_token_sweep.sh [SEED] [extra lerobot-train args...]

By default, supplement the completed 1/2/8/16 runs with 4 then 32 VLM prompt tokens,
sequentially on the same GPU(s). Set VLM_PROMPT_TOKEN_COUNTS to choose other lengths.
Each run starts from the SAME PRETRAINED_PATH, with action prompts=0, CABO=false,
no pretraining/bridge and 3000 flow updates by default. Stop on the first failure.

GPU_IDS=0 BATCH_SIZE=32 bash examples/training/train_pi05_vlm_token_sweep.sh 0
DRY_RUN=true bash examples/training/train_pi05_vlm_token_sweep.sh 0
VLM_PROMPT_TOKEN_COUNTS="4 32" bash examples/training/train_pi05_vlm_token_sweep.sh 0 \
  --wandb.enable=true --wandb.project=prompt-learning --wandb.mode=online --wandb.disable_artifact=true
VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" bash examples/training/train_pi05_vlm_token_sweep.sh 0

Overrides inherited from train_pi05_prompt_ablation.sh include DATASET_ROOT,
PRETRAINED_PATH, TOKENIZER_PATH, GPU_IDS, NUM_PROCESSES, BATCH_SIZE, FLOW_STEPS,
DTYPE, MIXED_PRECISION, SAVE_FREQ, SAVE_STEPS and DRY_RUN.

OUTPUT_ROOT defaults to /root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/vlm-token-sweep
LOG_ROOT defaults to /root/autodl-tmp/logs/prompt-learning/vlm-token-sweep
Run names: pi05-vlm_only-vlm{COUNT}-seedN; use RUN_PREFIX to change the prefix.
All destinations are checked before starting. Existing outputs are never overwritten
or silently skipped. After a partial sweep, explicitly select remaining token counts.
Use a base checkpoint without learned prompt banks; changing the length of a saved
learned prompt is deliberately rejected by the model loader.
EOF
  exit 0
fi

SEED="${1:-0}"
if (( $# )); then
  shift
fi
if [[ ! "${SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer without leading zeros" >&2
  exit 2
fi
EXTRA_ARGS=("$@")
# These arguments would invalidate the token-count labels or the vlm_only recipe.
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg%%=*}" in
    --policy.num_vlm_prompt_tokens|--policy.num_prompt_tokens|--policy.next_action_pretrain_steps|\
    --policy.next_action_bridge_steps|--policy.cabo_enabled|--output_dir|--job_name|--seed|--steps|--resume|--config_path|\
    --policy.path|--policy.pretrained_path|--policy.training_stage)
      echo "Sweep controls ${arg%%=*}; use the documented environment variables instead" >&2
      exit 2
      ;;
  esac
done
if [[ -n "${RUN_NAME:-}" || -n "${VLM_PROMPT_TOKENS:-}" ]]; then
  echo "For sweeps, use RUN_PREFIX and VLM_PROMPT_TOKEN_COUNTS instead of RUN_NAME/VLM_PROMPT_TOKENS" >&2
  exit 2
fi
read -r -a TOKEN_COUNTS <<< "${VLM_PROMPT_TOKEN_COUNTS-4 32}"
if (( ${#TOKEN_COUNTS[@]} == 0 )); then
  echo "VLM_PROMPT_TOKEN_COUNTS must contain at least one positive integer" >&2
  exit 2
fi
declare -A SEEN_COUNTS=()
for count in "${TOKEN_COUNTS[@]}"; do
  if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid VLM prompt token count: ${count}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_COUNTS[${count}]:-}" ]]; then
    echo "Duplicate VLM prompt token count: ${count}" >&2
    exit 2
  fi
  SEEN_COUNTS["${count}"]=1
done

export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/vlm-token-sweep}"
export LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/prompt-learning/vlm-token-sweep}"
RUN_PREFIX="${RUN_PREFIX:-pi05-vlm_only}"
DRY_RUN="${DRY_RUN:-false}"
run_one() {
  local count="$1" dry_run="$2"
  VLM_PROMPT_TOKENS="${count}" RUN_NAME="${RUN_PREFIX}-vlm${count}-seed${SEED}" DRY_RUN="${dry_run}" \
    bash "${SCRIPT_DIR}/train_pi05_prompt_ablation.sh" vlm_only "${SEED}" "${EXTRA_ARGS[@]}"
}

echo "Checking all ${#TOKEN_COUNTS[@]} runs: tokens=${TOKEN_COUNTS[*]} seed=${SEED}"
for count in "${TOKEN_COUNTS[@]}"; do
  run_one "${count}" true >/dev/null
done
for count in "${TOKEN_COUNTS[@]}"; do
  echo "Starting vlm_only: ${count} VLM prompt tokens"
  run_one "${count}" "${DRY_RUN}"
done
