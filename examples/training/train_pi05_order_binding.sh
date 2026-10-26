#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_order_binding.sh occ|slot [--wandb.*=...]
occ:  GPU0,1 / port29551; noise/time follows immutable sample-use IDs.
slot: GPU2,3 / port29552; the SAME indexed CPU bank follows global batch slots.
Each queue sequentially runs FLOW_BANKS="1 0" x WITHIN_SEEDS="0 1" (four models).
Each bank b means E=b,T=b jointly, not separate E/T main-effect estimates.
Both: base, P1/D0, token2, global seed0, batch32 PER GPU, 3000 steps, compile=false,
only final full checkpoint003000; full input traces and original fixed-probe diagnostics.
Names: pi05-orderbind-r1-W0-Bocc-base-vlm2-P1-D0-E1-T1 (Bslot for slot).

Controls: EXPERIMENT_ID=1 (name only), FLOW_BANKS, WITHIN_SEEDS, GPU_IDS,
MAIN_PROCESS_PORT, OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT, PROMPT_INIT_DIR,
BASE_PRETRAINED_PATH, TOKENIZER_PATH, PLAN_DIR, DRY_RUN=true.
Queues use separate default plan directories to avoid concurrent preparation writes;
their plans have IDENTICAL content/hashes. Do not share an unprepared PLAN_DIR.
All selected outputs must be absent. After interruption explicitly select only the
unstarted bank/order cells; do not silently skip or overwrite previous runs.
EOF
  exit 0
fi
MODE="${1:-}"; shift || true
case "$MODE" in
  occ) BINDING=occurrence; DEFAULT_GPUS=0,1; DEFAULT_PORT=29551 ;;
  slot) BINDING=slot; DEFAULT_GPUS=2,3; DEFAULT_PORT=29552 ;;
  *) echo 'Expected occ or slot; see --help' >&2; exit 2 ;;
esac
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this experiment controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export GPU_IDS="${GPU_IDS:-$DEFAULT_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$DEFAULT_PORT}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"; DRY_RUN="${DRY_RUN:-false}"
PROMPT_INIT_DIR="${PROMPT_INIT_DIR:-/data1/wyn/prompt-work}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
case "$DRY_RUN" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
PLAN_DIR="${PLAN_DIR:-/data1/wyn/prompt-work/orderbind-r${EXPERIMENT_ID}-${MODE}}"
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
RUN_DIRS=(); SELECTED_BANKS=(); SELECTED_ORDERS=()
for b in "${BANKS[@]}"; do
  for w in "${ORDERS[@]}"; do
    name="pi05-orderbind-r${EXPERIMENT_ID}-W${w}-B${MODE}-base-vlm2-P1-D0-E${b}-T${b}"
    [[ ! -e "$OUTPUT_ROOT/$name" ]] || { echo "Output already exists: $OUTPUT_ROOT/$name" >&2; exit 1; }
    RUN_DIRS+=("$OUTPUT_ROOT/$name"); SELECTED_BANKS+=("$b"); SELECTED_ORDERS+=("$w")
  done
done
LIBERO_PROMPT_INIT=$(python -m lerobot.scripts.export_pi05_prompt_init resolve-libero-init "$PROMPT_INIT_DIR" --seed 1)
python - "$LIBERO_PROMPT_INIT" <<'PY'
import sys
from lerobot.utils.prompt_initialization import read_initial_prompt
_, metadata = read_initial_prompt(sys.argv[1], 2)
if metadata["source_training_seed"] != 1 or metadata.get("source_prompt_init_path"):
    raise SystemExit("P1 must be an original, unswapped seed-1 initialization")
print("P1 initialization provenance OK")
PY
python -m lerobot.scripts.pdet_followup prepare --dataset-root "$DATASET_ROOT" \
  --output "$PLAN_DIR" --within-seeds "${ORDERS[@]}" --data-seed 0 --steps 3000 --batch-size 32 --world-size 2
WANDB_ARGS=("$@")
run_one() {
  local i="$1" dry="$2" b="${SELECTED_BANKS[$1]}" w="${SELECTED_ORDERS[$1]}"
  RUN_NAME="$(basename -- "${RUN_DIRS[$i]}")" VLM_PROMPT_TOKENS=2 DRY_RUN="$dry" \
  PROMPT_INIT_PATH="$LIBERO_PROMPT_INIT" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
    --policy.compile_model=false \
    --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
    --pdet.noise_seed="$b" --pdet.time_seed="$b" \
    --pdet.order_plan="$PLAN_DIR/D0-W${w}.npz" --pdet.flow_binding="$BINDING" \
    --pdet.audit_steps=3000 --pdet.diagnostics=true \
    --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
    --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
    --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
    --log_freq=10 "${WANDB_ARGS[@]}"
}
for i in "${!RUN_DIRS[@]}"; do run_one "$i" true >/dev/null; done
echo "All ${#RUN_DIRS[@]} preflights passed; binding=$BINDING GPUs=$GPU_IDS port=$MAIN_PROCESS_PORT"
for i in "${!RUN_DIRS[@]}"; do
  echo "Starting $(basename -- "${RUN_DIRS[$i]}")"
  run_one "$i" "$DRY_RUN"
  if [[ "$DRY_RUN" == false ]]; then
    python -m lerobot.scripts.pdet_followup summarize "${RUN_DIRS[$i]}"
  fi
done
if [[ "$DRY_RUN" == false ]]; then
  python -m lerobot.scripts.pdet_order_binding audit --allow-partial "${RUN_DIRS[@]}"
fi
