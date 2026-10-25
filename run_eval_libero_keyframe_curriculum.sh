#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_keyframe_curriculum.sh baseline|weight|curriculum [all|no10|SUITE]
baseline: B:0 B:1; weight: K2:0 K2:1 R2:0 R2:1;
curriculum: Kearly:0 Kearly:1 Klate:0 Klate:1 (only new cells, not earlier stages).
CELLS="B:0" can select/subdivide cells within this stage across GPUs.
Defaults: GPU_ID=0, EXPERIMENT_ID=1, checkpoint3000, EVAL_SEED=1000,
10 episodes/task, quiet logs + one live progress line and final success rates.
Same model/path controls as run_eval_libero_vlm_only_pdet_sweep.sh. DRY_RUN=true supported.
EOF
  exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
STAGE="${1:-}"; SUITES="${2:-all}"
case "$STAGE" in
  baseline) DEFAULT_CELLS='B:0 B:1' ;;
  weight) DEFAULT_CELLS='K2:0 K2:1 R2:0 R2:1' ;;
  curriculum) DEFAULT_CELLS='Kearly:0 Kearly:1 Klate:0 Klate:1' ;;
  *) echo 'Invalid stage; see --help' >&2; exit 2 ;;
esac
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; keyframe evaluator controls it" >&2; exit 2; fi
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export GPU_ID="${GPU_ID:-0}" CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
[[ "$CHECKPOINT_STEP" == 3000 ]] || { echo 'This experiment evaluates checkpoint003000' >&2; exit 2; }
read -r -a SELECTED <<< "${CELLS-$DEFAULT_CELLS}"
(( ${#SELECTED[@]} )) || { echo 'Empty CELLS selection' >&2; exit 2; }
declare -A SEEN=()
for cell in "${SELECTED[@]}"; do
  [[ " $DEFAULT_CELLS " == *" $cell "* && -z "${SEEN[$cell]:-}" ]] || { echo "Invalid/duplicate cell: $cell" >&2; exit 2; }
  SEEN[$cell]=1
  mode="${cell%:*}"; bank="${cell#*:}"
  name="pi05-key-r${EXPERIMENT_ID}-${mode}-W0-Bocc-base-vlm2-P1-D0-E${bank}-T${bank}"
  python -m lerobot.scripts.keyframe_experiment check-checkpoint "$CKPT_ROOT/$name" --step 3000 >/dev/null
done
for cell in "${SELECTED[@]}"; do
  mode="${cell%:*}"; bank="${cell#*:}"
  RUN_PREFIX="pi05-key-r${EXPERIMENT_ID}-${mode}-W0-Bocc" PDET_CASES="10${bank}${bank}" \
    bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
done
