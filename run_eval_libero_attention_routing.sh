#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_eval_libero_attention_routing.sh [all|no10|SUITE] [TRAINING_SEED]

Same vlm_only checkpoint 003000 as the action-attention launcher; no training.
Record the final action-expert layer, both cameras and ALL key groups, preserving
each head and each denoising pass. Queries remain the first n_action_steps action
tokens, averaged together. Individual action query rows are not retained.

GPU_ID=0 BASE_CKPT=/actual/run/checkpoints ATTENTION_FONT_PATH=/actual/times.ttf \
  bash run_eval_libero_attention_routing.sh libero_10 0

ATTENTION_LAYER=-1     final layer, or a zero-based layer index
EPISODES_PER_TASK=1    pilot; increase only after validating the routing schema
BASE_OUTPUT=...       use a fresh output directory for each changed setting

Outputs beside each attention video:
  .attention.npz                   maps/RGB + routing arrays
  .attention.routing.csv           per-head/pass group mass and token-count controls
  .attention.routing_summary.json  episode summary and per-prediction group masses
  .attention.routing_top_tokens.csv top 20 keys after display head/pass reduction

For the upstream VLM-prompt route, use ATTENTION_RECORD_ROUTING=1 with
run_eval_libero_vlm_prompt_attention.sh and a separate BASE_OUTPUT.
See examples/analysis/README_attention_routing.md for the experiment protocol.
HELP
  exit 0
fi
export ATTENTION_RECORD_ROUTING=1
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/2601-lerobot-attention-routing}"
exec bash "$SCRIPT_DIR/run_eval_libero_action_attention.sh" "$@"
