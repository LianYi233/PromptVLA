#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_order_binding.sh occ|slot|all [all|no10|SUITE]
Defaults: EXPERIMENT_ID=1, FLOW_BANKS="1 0", WITHIN_SEEDS="0 1", checkpoint3000,
EVAL_SEED=1000, 10 episodes/task, single-line progress + final success rates.
occ defaults GPU0, slot GPU2, all GPU0. Run occ/slot in separate terminals to parallelize.
Names/config bindings are checked before the standard PDET checkpoint/evaluation preflight.
Same CKPT_ROOT/BASE_OUTPUT/LOG_ROOT and model/tokenizer overrides as the PDET evaluator.
Use DRY_RUN=true to validate without GPU evaluation. No waiting by default.
EOF
  exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
MODE="${1:-}"; SUITES="${2:-all}"
case "$MODE" in
  occ) MODES=(occ); DEFAULT_GPU=0 ;;
  slot) MODES=(slot); DEFAULT_GPU=2 ;;
  all) MODES=(occ slot); DEFAULT_GPU=0 ;;
  *) echo 'Expected occ, slot or all; see --help' >&2; exit 2 ;;
esac
case "$SUITES" in all|no10|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite' >&2; exit 2 ;; esac
for key in PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; order-binding evaluator controls it" >&2; exit 2; fi
done
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export GPU_ID="${GPU_ID:-$DEFAULT_GPU}" CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export WAIT_FOR_CHECKPOINTS=false PROGRESS="${PROGRESS:-true}"
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
[[ "$CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CHECKPOINT_STEP' >&2; exit 2; }
read -r -a BANKS <<< "${FLOW_BANKS-1 0}"
read -r -a ORDERS <<< "${WITHIN_SEEDS-0 1}"
(( ${#BANKS[@]} && ${#ORDERS[@]} )) || { echo 'Empty bank/order selection' >&2; exit 2; }
declare -A SEEN_BANKS=() SEEN_ORDERS=()
for b in "${BANKS[@]}"; do
  [[ "$b" =~ ^[01]$ && -z "${SEEN_BANKS[$b]:-}" ]] || { echo "Invalid/duplicate bank: $b" >&2; exit 2; }
  SEEN_BANKS[$b]=1
done
for w in "${ORDERS[@]}"; do
  [[ "$w" =~ ^(0|[1-9][0-9]*)$ && -z "${SEEN_ORDERS[$w]:-}" ]] || { echo "Invalid/duplicate W: $w" >&2; exit 2; }
  SEEN_ORDERS[$w]=1
done
for binding in "${MODES[@]}"; do
  for b in "${BANKS[@]}"; do
    for w in "${ORDERS[@]}"; do
      prefix="pi05-orderbind-r${EXPERIMENT_ID}-W${w}-B${binding}"
      name="${prefix}-base-vlm2-P1-D0-E${b}-T${b}"
      python -m lerobot.scripts.pdet_order_binding check-checkpoint "$CKPT_ROOT/$name" \
        --step "$CHECKPOINT_STEP" >/dev/null
      RUN_PREFIX="$prefix" PDET_CASES="10${b}${b}" \
        bash "$SCRIPT_DIR/run_eval_libero_vlm_only_pdet_sweep.sh" "$SUITES"
    done
  done
done
