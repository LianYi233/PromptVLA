#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_base.sh [all|no10|libero_spatial|libero_object|libero_goal|libero_10]

Evaluate the starting pi05_libero_base weights with BOTH prompt banks disabled.
No training, no random prompt insertion, no checkpoint conversion.
Defaults match the token sweep: 10 episodes/task, n_action_steps=10, evaluation
seed=1000, CUDA GPU 0, AMP/compile disabled, task-level resume and success summaries.

GPU_ID=0 bash run_eval_libero_base.sh all
DRY_RUN=true bash run_eval_libero_base.sh all

BASE_MODEL_PATH defaults to PRETRAINED_PATH, or /root/autodl-tmp/models/pi05_libero_base
TOKENIZER_PATH defaults to /root/autodl-tmp/models/google/paligemma-3b-pt-224
BASE_OUTPUT defaults to /root/autodl-tmp/eval/prompt-learning/pi05-libero-base
Other overrides: GPU_ID, EPISODES_PER_TASK, LIBERO_CONFIG_PATH.
Uses the base model's saved processors/statistics; only the tokenizer path is
overridden at runtime. The checkpoint directory and its JSON files are not edited.
EOF
  exit 0
fi
if (( $# > 1 )); then
  echo "Expected [all|no10|SUITE]; the base model has no training-run seed to select" >&2
  exit 2
fi
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_base}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
exec bash "$SCRIPT_DIR/run_eval_libero-full.sh" pi05_libero_base "${1:-all}" 0
