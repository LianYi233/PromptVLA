#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_frameskip.sh [all|no10|SUITE]
Sequential final-checkpoint evaluation of U/R/S/C x E0T0/E1T1.
CELLS="U0 U1 R0 R1 S0 S1 C0 C1"; EXPERIMENT_ID=1; GPU_ID=2.
Uses the standard PDET evaluator: seed1000, 10 episodes/task, execute10 actions,
single-line progress and final rates. Controls: CKPT_ROOT, BASE_OUTPUT, LOG_ROOT,
BASE_PRETRAINED_PATH, TOKENIZER_PATH, DRY_RUN=true, PROGRESS=true.
EOF
  exit 0
fi
(( $# <= 1 )) || { echo 'Too many arguments' >&2; exit 2; }
SUITES="${1:-all}"
case "$SUITES" in all|no10|libero_10|libero_goal|libero_object|libero_spatial) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  [[ -z "${!key:-}" ]] || { echo "Unset $key" >&2; exit 2; }
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export GPU_ID="${GPU_ID:-2}" CHECKPOINT_STEP=3000
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
read -r -a REQUESTED <<< "${CELLS-U0 U1 R0 R1 S0 S1 C0 C1}"
(( ${#REQUESTED[@]} )) || { echo 'Empty CELLS' >&2; exit 2; }
declare -A SEEN=()
for cell in "${REQUESTED[@]}"; do
  [[ "$cell" =~ ^[URSC][01]$ && -z "${SEEN[$cell]:-}" ]] || { echo "Invalid/duplicate cell: $cell" >&2; exit 2; }
  SEEN[$cell]=1
done
for cell in "${REQUESTED[@]}"; do
  b="${cell:1:1}"
  prefix="pi05-fskip-r${EXPERIMENT_ID}-${cell:0:1}-Bocc"
  name="${prefix}-base-vlm2-P1-D0-E${b}-T${b}"
  python -m lerobot.scripts.frameskip_experiment check-checkpoint "$CKPT_ROOT/$name" --step 3000 >/dev/null
  RUN_PREFIX="$prefix" PDET_CASES="10${b}${b}" \
    bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
done
