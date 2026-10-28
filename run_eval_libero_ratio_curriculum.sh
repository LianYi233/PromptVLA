#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_ratio_curriculum.sh [all|no10|SUITE]
Use the training FLOW_STEPS, CURRICULUM_DIRECTION, EASY_PERCENT, SAMPLE_SEED, TRAIN_SCOPE and REPEAT_ID.
Defaults: easy_to_hard/e70 (hard_to_easy defaults to e30), S0/r1, GPU_ID=0,
FLOW_STEPS=6000/checkpoint006000, EVAL_SEED=1000, EPISODES_PER_TASK=10, PROGRESS=true.
Set FLOW_STEPS=3000 to evaluate the legacy 3000-step names/checkpoint003000.
Controls: SOURCE_PAIR_DIR, ORDER_DIR, CKPT_ROOT, BASE_OUTPUT, EVAL_LOG_ROOT, DRY_RUN.
TRAIN_SCOPE=joint|expert_only|prompt_only (default joint); random is also supported.
Always validates the declared scope and loads the complete trained checkpoint.
EOF
  exit 0
fi
(( $# <= 1 )) || { echo 'Too many arguments' >&2; exit 2; }
SUITES="${1:-all}"
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this evaluator controls it" >&2; exit 2; fi
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
FLOW_STEPS="${FLOW_STEPS:-6000}"
[[ "$FLOW_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid FLOW_STEPS' >&2; exit 2; }
if [[ -n "${CHECKPOINT_STEP:-}" && "$CHECKPOINT_STEP" != "$FLOW_STEPS" ]]; then
  echo 'CHECKPOINT_STEP must equal FLOW_STEPS; only the final checkpoint is saved' >&2; exit 2
fi
export GPU_ID="${GPU_ID:-0}" CHECKPOINT_STEP="$FLOW_STEPS"
TRAIN_SCOPE="${TRAIN_SCOPE:-joint}"
case "$TRAIN_SCOPE" in
  joint) SCOPE_TAG=aexpert; export TRAIN_ACTION_EXPERT=true ;;
  expert_only) SCOPE_TAG=frozenp-aexpert; export TRAIN_ACTION_EXPERT=true ;;
  prompt_only) SCOPE_TAG=promptonly; export TRAIN_ACTION_EXPERT=false ;;
  *) echo 'Invalid TRAIN_SCOPE' >&2; exit 2 ;;
esac
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
REPEAT_ID="${REPEAT_ID:-1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
CURRICULUM_DIRECTION="${CURRICULUM_DIRECTION:-easy_to_hard}"
case "$CURRICULUM_DIRECTION" in
  easy_to_hard) EASY_PERCENT="${EASY_PERCENT:-70}" ;;
  hard_to_easy) EASY_PERCENT="${EASY_PERCENT:-30}" ;;
  random) EASY_PERCENT="${EASY_PERCENT:-50}" ;;
  *) echo 'Invalid CURRICULUM_DIRECTION' >&2; exit 2 ;;
esac
SAMPLE_SEED="${SAMPLE_SEED:-0}"
LABEL=$(python -m lerobot.scripts.pi05_ratio_curriculum label \
  --direction "$CURRICULUM_DIRECTION" --easy-percent "$EASY_PERCENT" --sample-seed "$SAMPLE_SEED" --steps "$FLOW_STEPS")
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
ORDER_DIR="${ORDER_DIR:-/data1/wyn/prompt-work/ratio-curriculum/$LABEL}"
prefix="pi05-curriculum-${LABEL}-${SCOPE_TAG}-r${REPEAT_ID}-libero-Bocc"
python -m lerobot.scripts.pi05_ratio_curriculum check-checkpoint \
  "$ORDER_DIR" "$CKPT_ROOT/${prefix}-base-vlm2-P1-D0-E1-T1" >/dev/null
python - "$ORDER_DIR/curriculum.json" "$SOURCE_PAIR_DIR" <<'PY'
import json, sys
from pathlib import Path
if Path(json.loads(Path(sys.argv[1]).read_text())["pair"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("SOURCE_PAIR_DIR differs from the prepared curriculum")
PY
BASE_PRETRAINED_PATH="$SOURCE_PAIR_DIR/libero" RUN_PREFIX="$prefix" PDET_CASES=1011 \
  bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
