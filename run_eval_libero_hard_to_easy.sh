#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash run_eval_libero_hard_to_easy.sh [all|no10|SUITE]'
  echo 'Defaults: GPU_ID=0 REPEAT_ID=1 checkpoint003000 EVAL_SEED=1000 EPISODES_PER_TASK=10.'
  echo 'Controls: SOURCE_PAIR_DIR, ORDER_DIR, CKPT_ROOT, BASE_OUTPUT, EVAL_LOG_ROOT, PROGRESS, DRY_RUN.'
  echo 'TRAIN_ACTION_EXPERT=true selects the full-action-network run; default false is prompt-only.'
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
export GPU_ID="${GPU_ID:-0}" CHECKPOINT_STEP=3000
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
REPEAT_ID="${REPEAT_ID:-1}"
export TRAIN_ACTION_EXPERT="${TRAIN_ACTION_EXPERT:-false}"
case "$TRAIN_ACTION_EXPERT" in
  false) EXPERT_TAG='' ;;
  true) EXPERT_TAG='aexpert-' ;;
  *) echo 'Invalid TRAIN_ACTION_EXPERT' >&2; exit 2 ;;
esac
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
ORDER_DIR="${ORDER_DIR:-/data1/wyn/prompt-work/hard2easy-r1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
prefix="pi05-hard2easy-${EXPERT_TAG}r${REPEAT_ID}-libero-Bocc"
python -m lerobot.scripts.pi05_hard_to_easy check-checkpoint \
  "$ORDER_DIR" "$CKPT_ROOT/${prefix}-base-vlm2-P1-D0-E1-T1" >/dev/null
python - "$ORDER_DIR/curriculum.json" "$SOURCE_PAIR_DIR" <<'PY'
import json, sys
from pathlib import Path
if Path(json.loads(Path(sys.argv[1]).read_text())["pair"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("SOURCE_PAIR_DIR differs from the prepared curriculum")
PY
BASE_PRETRAINED_PATH="$SOURCE_PAIR_DIR/libero" RUN_PREFIX="$prefix" PDET_CASES=1011 \
  bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
