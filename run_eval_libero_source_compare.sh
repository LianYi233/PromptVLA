#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_source_compare.sh libero|generic|both [all|no10|SUITE]
Defaults: REPEAT_ID=1, checkpoint003000, EVAL_SEED=1000, 10 episodes/task,
one updating progress line and final rates. Single GPU/model; both is sequential.
Controls: SOURCE_PAIR_DIR, CKPT_ROOT, BASE_OUTPUT, EVAL_LOG_ROOT, GPU_ID,
TOKENIZER_PATH, DRY_RUN. Source and run-name overrides are rejected.
EOF
  exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
case "${1:-}" in
  libero) SOURCES=(libero); DEFAULT_GPU=0 ;;
  generic) SOURCES=(generic); DEFAULT_GPU=2 ;;
  both) SOURCES=(libero generic); DEFAULT_GPU=0 ;;
  *) echo 'Expected libero, generic or both; see --help' >&2; exit 2 ;;
esac
SUITES="${2:-all}"
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; source comparison controls it" >&2; exit 2; fi
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export GPU_ID="${GPU_ID:-$DEFAULT_GPU}" CHECKPOINT_STEP=3000
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
REPEAT_ID="${REPEAT_ID:-1}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
for source in "${SOURCES[@]}"; do
  prefix="pi05-source-r${REPEAT_ID}-${source}-W0-Bocc"
  name="${prefix}-base-vlm2-P1-D0-E1-T1"
  python -m lerobot.scripts.pi05_source_compare check-checkpoint "$SOURCE_PAIR_DIR" "$CKPT_ROOT/$name" >/dev/null
  BASE_PRETRAINED_PATH="$SOURCE_PAIR_DIR/$source" RUN_PREFIX="$prefix" PDET_CASES=1011 \
    bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
done
