#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash run_eval_libero_two_frame.sh E2H|H2E [all|no10|SUITE]'
  echo 'Defaults: PAIR_DIR=/data1/wyn/prompt-work/twoframe-r1 FLOW_STEPS=6000 REPEAT_ID=1 GPU_ID=0.'
  echo 'Four suites, seed1000, 10episodes/task; trained VLM2 + full action expert, final checkpoint only.'
  exit 0
fi
(( $# >= 1 && $# <= 2 )) || { echo 'Choose E2H/H2E and optional suite' >&2; exit 2; }
DIRECTION="$1"; SUITES="${2:-all}"
case "$DIRECTION" in E2H|H2E) ;; *) echo 'Invalid direction' >&2; exit 2 ;; esac
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this evaluator controls it" >&2; exit 2; fi
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
FLOW_STEPS="${FLOW_STEPS:-6000}"; REPEAT_ID="${REPEAT_ID:-1}"
for key in FLOW_STEPS REPEAT_ID; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $key" >&2; exit 2; }
done
if [[ -n "${CHECKPOINT_STEP:-}" && "$CHECKPOINT_STEP" != "$FLOW_STEPS" ]]; then
  echo 'CHECKPOINT_STEP must equal FLOW_STEPS' >&2; exit 2
fi
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export GPU_ID="${GPU_ID:-0}" CHECKPOINT_STEP="$FLOW_STEPS" TRAIN_ACTION_EXPERT=true
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
PAIR_DIR="${PAIR_DIR:-/data1/wyn/prompt-work/twoframe-r1}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
PREFIX=$(python -m lerobot.scripts.pi05_two_frame prefix "$PAIR_DIR" "$DIRECTION" --repeat "$REPEAT_ID" --steps "$FLOW_STEPS")
python -m lerobot.scripts.pi05_two_frame check-checkpoint "$PAIR_DIR" "$CKPT_ROOT/${PREFIX}-base-vlm2-P1-D0-E1-T1" >/dev/null
python - "$PAIR_DIR/pair.json" "$SOURCE_PAIR_DIR" <<'PY'
import json, sys
from pathlib import Path
if Path(json.loads(Path(sys.argv[1]).read_text())["source_pair"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("SOURCE_PAIR_DIR differs from prepared two-frame pair")
PY
BASE_PRETRAINED_PATH="$SOURCE_PAIR_DIR/libero" RUN_PREFIX="$PREFIX" PDET_CASES=1011 \
  bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
