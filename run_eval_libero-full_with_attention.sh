#!/usr/bin/env bash
set -euo pipefail

# Reuse standard evaluation, CUDA checks, progress, success metrics, and resume.
# Defaults: action-token queries, final expert layer, first policy camera,
# average over all heads and flow denoising passes. No extra model forward.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_eval_libero-full_with_attention.sh VARIANT [all|no10] [SEED]
Example: GPU_ID=0 bash run_eval_libero-full_with_attention.sh full_reference all 0

Existing overrides: BASE_CKPT, CKPT_ROOT, BASE_OUTPUT, LIBERO_CONFIG_PATH, GPU_ID.
Attention overrides:
  ATTENTION_SOURCE=action      action | vlm_prompt (requires VLM prompts)
  ATTENTION_LAYER=-1           zero-based layer; negative values count from end
  ATTENTION_CAMERA=0           index in policy image order (0=main, 1=wrist normally)
  ATTENTION_CAMERAS=0,1        optional two-camera grid; overrides ATTENTION_CAMERA
  ATTENTION_SNAPSHOT_EVERY=0   with two cameras, save PNGs every N predictions; 0=off
  ATTENTION_DENOISE=mean       mean | first | last (VLM prefix runs only once)
  ATTENTION_ALPHA=0.82         overlay opacity in [0,1]
  ATTENTION_VMAX=0             0=per-prediction relative; >0=fixed probability scale
  ATTENTION_FONT_PATH=...      optional path to Times New Roman .ttf (auto-detected)
  ATTENTION_RECORD_ROUTING=1   additionally save all-key masses per head/pass (default 0)
  EPISODES_PER_TASK=10         positive episode count per task; suite name also accepted as MODE

Overlay: blue/purple low attention, bright red/yellow high attention.
All labels use Times New Roman. Missing fonts cause an explicit error.

All 10 episodes/task are saved, including failures. Each video shows live view,
the exact policy input frame, and its heatmap. Queued actions hold the source
frame and map together. Raw patch probabilities and settings are saved alongside
each video as .attention.npz / .attention.json. No weights are changed.
Default output root: /root/autodl-tmp/eval/2601-lerobot-attention
Use a new BASE_OUTPUT when changing settings; resume validates configuration.
HELP
  exit 0
fi
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LEROBOT_EVAL_ATTENTION=1
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/eval/2601-lerobot-attention}"
exec bash "$SCRIPT_DIR/run_eval_libero-full.sh" "$@"
