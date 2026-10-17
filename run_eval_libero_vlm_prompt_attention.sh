#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_eval_libero_vlm_prompt_attention.sh [all|no10|libero_goal|libero_10|libero_object|libero_spatial] [SEED]

Default: vlm_only checkpoint 003000, all four suites, one episode/task,
last VLM layer, cameras 0 and 1 captured in the SAME prediction.
The training variant called vlm_prompt_only in discussions is named vlm_only in this repo.

GPU_ID=0 BASE_CKPT=/actual/run/checkpoints ATTENTION_FONT_PATH=/actual/times.ttf \
  bash run_eval_libero_vlm_prompt_attention.sh libero_10 0

EPISODES_PER_TASK=10 restores the standard 100 episodes/suite evaluation.
ATTENTION_LAYER=-1 selects the final VLM layer (zero-based indices also accepted).
ATTENTION_SNAPSHOT_EVERY=5 saves PNGs every five predictions plus the last; 0 disables PNGs.
ATTENTION_VMAX=0 uses a shared per-prediction scale across cameras; >0 fixes the scale across time.
ATTENTION_CAMERAS=0,1 overrides the two policy-camera indices/order.
BASE_OUTPUT defaults to /root/autodl-tmp/eval/2601-lerobot-vlm-prompt-two-cameras.
Use a fresh BASE_OUTPUT after changing settings. No training is required.

Outputs: synchronized 2x2 PNGs, video, raw maps + lossless source RGB in NPZ,
and per-camera attention masses in CSV. See examples/analysis/README_libero_attention.md.
HELP
  exit 0
fi
if (( $# > 2 )); then
  echo "Expected [SUITE|all|no10] [SEED]; use --help" >&2
  exit 2
fi
export ATTENTION_SOURCE=vlm_prompt
export ATTENTION_CAMERAS="${ATTENTION_CAMERAS:-0,1}"
export ATTENTION_SNAPSHOT_EVERY="${ATTENTION_SNAPSHOT_EVERY:-5}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/2601-lerobot-vlm-prompt-two-cameras}"
exec bash "$SCRIPT_DIR/run_eval_libero-full_with_attention.sh" vlm_only "${1:-all}" "${2:-0}"
