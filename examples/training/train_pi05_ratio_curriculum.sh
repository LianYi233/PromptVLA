#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_ratio_curriculum.sh [--wandb.*=...]
Train 2 VLM prompt tokens + the FULL action expert, starting from pi05_libero_base.
CURRICULUM_DIRECTION=easy_to_hard|hard_to_easy (default easy_to_hard).
EASY_PERCENT=1..99: total EASY exposure percent, NOT a per-batch mixture or loss weight.
Default 70 for easy_to_hard, 30 for hard_to_easy. Hard percent = 100 - EASY_PERCENT.
Each task's candidate frames are split into lower/upper score halves. Resample each
group to its quota, then sort within fixed task slots. Repeat uses get fresh IDs/E/T.
SAMPLE_SEED=0 controls group sampling only; P1/D0/E1/T1 remain fixed.
Fixed: TRAIN_ACTION_EXPERT=true, VLM2, global seed0, 2 GPUs x batch32, 3000 steps,
bf16, no compile, no loss weighting, final-only checkpoint003000.
Defaults: GPU_IDS=0,1 MAIN_PROCESS_PORT=29584 REPEAT_ID=1 DRY_RUN=false.
SOURCE_PAIR_DIR=/data1/wyn/prompt-work/source-pair-r1
SCORES_PATH=/data1/wyn/prompt-work/frameskip-r1/scores.npz
ORDER_DIR defaults to /data1/wyn/prompt-work/ratio-curriculum/<direction-ratio-seed>.
Other controls: REFERENCE_RUN, OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT, TOKENIZER_PATH.
CPU preparation is automatic/idempotent; no rescoring or overwriting model outputs.
DRY_RUN=true prepares the plan and prints the command; no model/GPU is loaded.
EOF
  exit 0
fi
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this experiment controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export GPU_IDS="${GPU_IDS:-0,1}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29584}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
if [[ "${TRAIN_ACTION_EXPERT:-true}" != true ]]; then
  echo 'This recipe requires TRAIN_ACTION_EXPERT=true' >&2; exit 2
fi
export TRAIN_ACTION_EXPERT=true
REPEAT_ID="${REPEAT_ID:-1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
case "${DRY_RUN:-false}" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
CURRICULUM_DIRECTION="${CURRICULUM_DIRECTION:-easy_to_hard}"
case "$CURRICULUM_DIRECTION" in
  easy_to_hard) EASY_PERCENT="${EASY_PERCENT:-70}" ;;
  hard_to_easy) EASY_PERCENT="${EASY_PERCENT:-30}" ;;
  *) echo 'Invalid CURRICULUM_DIRECTION' >&2; exit 2 ;;
esac
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SETTINGS=(--direction "$CURRICULUM_DIRECTION" --easy-percent "$EASY_PERCENT" --sample-seed "$SAMPLE_SEED")
LABEL=$(python -m lerobot.scripts.pi05_ratio_curriculum label "${SETTINGS[@]}")
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
ORDER_DIR="${ORDER_DIR:-/data1/wyn/prompt-work/ratio-curriculum/$LABEL}"
SCORES_PATH="${SCORES_PATH:-/data1/wyn/prompt-work/frameskip-r1/scores.npz}"
REFERENCE_RUN="${REFERENCE_RUN:-$OUTPUT_ROOT/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1}"
export RUN_NAME="pi05-curriculum-${LABEL}-aexpert-r${REPEAT_ID}-libero-Bocc-base-vlm2-P1-D0-E1-T1"
[[ ! -e "$OUTPUT_ROOT/$RUN_NAME" ]] || { echo "Output already exists: $OUTPUT_ROOT/$RUN_NAME" >&2; exit 1; }
python -m lerobot.scripts.pi05_ratio_curriculum prepare \
  --pair "$SOURCE_PAIR_DIR" --scores "$SCORES_PATH" --reference-run "$REFERENCE_RUN" \
  --output "$ORDER_DIR" "${SETTINGS[@]}"
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
  --pdet.order_plan="$ORDER_DIR/plan.npz" --pdet.flow_binding=occurrence \
  --pdet.audit_steps=3000 --pdet.diagnostics=true \
  --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
  --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
  --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
  --log_freq=10 "$@"
if [[ "${DRY_RUN:-false}" == false ]]; then
  cp -- "$ORDER_DIR/curriculum.json" "$OUTPUT_ROOT/$RUN_NAME/difficulty_curriculum.json"
  python -m lerobot.scripts.pi05_ratio_curriculum audit "$ORDER_DIR" "$OUTPUT_ROOT/$RUN_NAME"
fi
