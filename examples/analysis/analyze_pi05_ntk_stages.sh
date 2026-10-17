#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash $0 RUN_DIR [GPU_ID] [extra analysis arguments...]" >&2
  exit 2
fi
RUN_DIR="$1"
GPU_ID="${2:-0}"
if (( $# >= 2 )); then shift 2; else shift; fi
ARGS=(
  --run-dir="${RUN_DIR}"
  --final-flow-steps="${FINAL_FLOW_STEPS:-2000}"
  --dataset-repo-id="${DATASET_REPO_ID:-libero}"
  --dataset-root="${DATASET_ROOT:-/root/autodl-tmp/datasets/libero}"
  --output-dir="${NTK_OUTPUT_DIR:-${RUN_DIR}/ntk_stages}"
)
if [[ -n "${TOKENIZER_PATH:-}" ]]; then ARGS+=(--tokenizer-path="${TOKENIZER_PATH}"); fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m lerobot.scripts.analyze_pi05_ntk_stages "${ARGS[@]}" "$@"
