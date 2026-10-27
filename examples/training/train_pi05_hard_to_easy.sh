#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_hard_to_easy.sh [--wandb.*=...]
One pi05_libero_base run; exact P1/D0/E1/T1/Bocc source-comparison recipe.
Change only the original W0 occurrence order: descending cached score WITHIN each task.
No new scores, resampling, data warmup, loss weighting, prompt initialization or noise bank.
Fixed: VLM2, global seed0, 2 GPUs x batch32, 3000 steps, bf16, no compile, final-only.
Defaults: GPU_IDS=0,1 MAIN_PROCESS_PORT=29583 REPEAT_ID=1 DRY_RUN=false.
TRAIN_ACTION_EXPERT=true also unfreezes the full action transformer/maps/time MLPs;
uses a distinct pi05-hard2easy-aexpert-r* name. Default false preserves prompt-only.
SOURCE_PAIR_DIR=/data1/wyn/prompt-work/source-pair-r1
SCORES_PATH=/data1/wyn/prompt-work/frameskip-r1/scores.npz
ORDER_DIR=/data1/wyn/prompt-work/hard2easy-r1 (shared across repeats)
REFERENCE_RUN defaults to completed pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1.
Other controls: OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT, TOKENIZER_PATH.
CPU preparation is automatic/idempotent; existing model outputs are never overwritten.
DRY_RUN=true prepares the small plan and prints the training command without training.
EOF
  exit 0
fi
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this comparison controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export GPU_IDS="${GPU_IDS:-0,1}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29583}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
REPEAT_ID="${REPEAT_ID:-1}"
export TRAIN_ACTION_EXPERT="${TRAIN_ACTION_EXPERT:-false}"
case "$TRAIN_ACTION_EXPERT" in
  false) EXPERT_TAG='' ;;
  true) EXPERT_TAG='aexpert-' ;;
  *) echo 'Invalid TRAIN_ACTION_EXPERT' >&2; exit 2 ;;
esac
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
ORDER_DIR="${ORDER_DIR:-/data1/wyn/prompt-work/hard2easy-r1}"
SCORES_PATH="${SCORES_PATH:-/data1/wyn/prompt-work/frameskip-r1/scores.npz}"
REFERENCE_RUN="${REFERENCE_RUN:-$OUTPUT_ROOT/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
case "${DRY_RUN:-false}" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
export RUN_NAME="pi05-hard2easy-${EXPERT_TAG}r${REPEAT_ID}-libero-Bocc-base-vlm2-P1-D0-E1-T1"
[[ ! -e "$OUTPUT_ROOT/$RUN_NAME" ]] || { echo "Output already exists: $OUTPUT_ROOT/$RUN_NAME" >&2; exit 1; }
python -m lerobot.scripts.pi05_hard_to_easy prepare \
  --pair "$SOURCE_PAIR_DIR" --scores "$SCORES_PATH" --reference-run "$REFERENCE_RUN" --output "$ORDER_DIR"
python - "$SOURCE_PAIR_DIR/pair.json" "$DATASET_ROOT" <<'PY'
import json, sys
from pathlib import Path
if Path(json.loads(Path(sys.argv[1]).read_text())["dataset_root"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("DATASET_ROOT differs from the prepared comparison")
PY
PRETRAINED_PATH="$SOURCE_PAIR_DIR/libero" VLM_PROMPT_TOKENS=2 \
PROMPT_INIT_PATH="$SOURCE_PAIR_DIR/P1.safetensors" \
  bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
  --policy.compile_model=false \
  --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
  --pdet.noise_seed=1 --pdet.time_seed=1 \
  --pdet.order_plan="$ORDER_DIR/H2E.npz" --pdet.flow_binding=occurrence \
  --pdet.audit_steps=3000 --pdet.diagnostics=true \
  --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
  --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
  --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
  --log_freq=10 "$@"
if [[ "${DRY_RUN:-false}" == false ]]; then
  cp -- "$ORDER_DIR/curriculum.json" "$OUTPUT_ROOT/$RUN_NAME/difficulty_curriculum.json"
  python -m lerobot.scripts.pi05_hard_to_easy audit "$ORDER_DIR" "$OUTPUT_ROOT/$RUN_NAME"
fi
