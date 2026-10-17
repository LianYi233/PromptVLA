#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_eval_libero_action_attention.sh [all|no10|libero_goal|libero_10|libero_object|libero_spatial] [SEED]

Default: the SAME vlm_only checkpoint 003000 used by the VLM-prompt launcher,
one episode/task, action-expert FINAL Transformer layer, cameras 0 and 1.
Queries are the action tokens to be executed (first n_action_steps in the chunk),
not action prompts. Average all heads and, by default, all denoising passes.

GPU_ID=0 BASE_CKPT=/actual/run/checkpoints ATTENTION_FONT_PATH=/actual/times.ttf \
  bash run_eval_libero_action_attention.sh libero_10 0

ATTENTION_LAYER=-1           final action-expert layer; zero-based indices also accepted
ATTENTION_DENOISE=mean       mean | first | last; denoising pass is distinct from layer
ATTENTION_CAMERAS=0,1        both cameras in the same prediction, shared color scale
ATTENTION_SNAPSHOT_EVERY=5   PNG every N predictions plus the last; 0 disables PNGs
ATTENTION_VMAX=0             shared per-prediction maximum; >0 fixes the scale across time
EPISODES_PER_TASK=1          set 10 for the standard 100 episodes/suite evaluation

Output root: /root/autodl-tmp/eval/2601-lerobot-action-two-cameras
Use a fresh BASE_OUTPUT after changing layer/denoising settings. Previous prompt
recordings cannot be converted into action attention; run inference again.
Outputs: two-camera 2x2 videos and PNGs, raw NPZ maps/RGB, metadata, camera mass CSV.
No training is required. See examples/analysis/README_libero_attention.md.
HELP
  exit 0
fi
if (( $# > 2 )); then
  echo "Expected [SUITE|all|no10] [SEED]; use --help" >&2
  exit 2
fi
export ATTENTION_SOURCE=action
export ATTENTION_LAYER="${ATTENTION_LAYER:--1}"
export ATTENTION_DENOISE="${ATTENTION_DENOISE:-mean}"
export ATTENTION_CAMERAS="${ATTENTION_CAMERAS:-0,1}"
export ATTENTION_SNAPSHOT_EVERY="${ATTENTION_SNAPSHOT_EVERY:-5}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/2601-lerobot-action-two-cameras}"
exec bash "$SCRIPT_DIR/run_eval_libero-full_with_attention.sh" vlm_only "${1:-all}" "${2:-0}"
