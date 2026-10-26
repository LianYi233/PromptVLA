#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_pdet_followup.sh repeats|order [--wandb.*=...]
repeats: GPU0,1 / port29541, sequential 1011 0011 1001 1100, pi05-pdet-rep1-...
order:   GPU2,3 / port29542, sequential W0 W1 at P1D0E1T1, pi05-taskfixed-rep1-W0-...
Both: token=2, global seed=0, batch32 PER GPU, 3000 steps, only full checkpoint003000.
Replicate number is a NAME, never a new RNG seed. No overwrites or automatic skips.
Order uses a NEW occurrence-keyed E/T protocol. W0/W1 are random within-task permutations,
not easy/hard labels. Both preserve the complete pool and every task slot (including DDP).
Defaults record all3000 input batches, update/gradient norms every log window(10 steps),
rank/NTK and fixed TRAINING-probe loss at0,50,100,250,500,...,3000. No held-out claim.

Controls: REPEAT_ID=1, PDET_CASES(repeats only), WITHIN_SEEDS='0 1'(order only),
GPU_IDS, MAIN_PROCESS_PORT, PROMPT_INIT_DIR, OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT,
BASE_PRETRAINED_PATH, TOKENIZER_PATH, PLAN_DIR, DRY_RUN=true.
DRY_RUN builds/validates CPU plans and all launch commands, but never loads a model/trains.
After a partial queue: set PDET_CASES or WITHIN_SEEDS to only the unstarted runs.
EOF
  exit 0
fi
MODE="${1:-}"; shift || true
case "$MODE" in repeats|order) ;; *) echo 'Expected repeats or order; see --help' >&2; exit 2 ;; esac
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH RUN_NAME VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE RUN_PREFIX; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; follow-up controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
PROMPT_INIT_DIR="${PROMPT_INIT_DIR:-/data1/wyn/prompt-work}"
REPEAT_ID="${REPEAT_ID:-1}"; DRY_RUN="${DRY_RUN:-false}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
case "$DRY_RUN" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
declare -a LABELS=() CASES=() PLANS=() RUN_DIRS=()
if [[ "$MODE" == repeats ]]; then
  export GPU_IDS="${GPU_IDS:-0,1}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
  read -r -a REQUESTED <<< "${PDET_CASES-1011 0011 1001 1100}"
  for item in "${REQUESTED[@]}"; do
    [[ "$item" =~ ^[01]{4}$ ]] || { echo "Invalid PDET case: $item" >&2; exit 2; }
    CASES+=("$item"); LABELS+=("pi05-pdet-rep${REPEAT_ID}"); PLANS+=("")
  done
else
  if [[ -n "${PDET_CASES:-}" ]]; then echo 'Unset PDET_CASES; order uses fixed1011' >&2; exit 2; fi
  export GPU_IDS="${GPU_IDS:-2,3}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29542}"
  read -r -a REQUESTED <<< "${WITHIN_SEEDS-0 1}"
  PLAN_DIR="${PLAN_DIR:-/data1/wyn/prompt-work/pdet-order-rep${REPEAT_ID}}"
  for w in "${REQUESTED[@]}"; do
    [[ "$w" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid W seed: $w" >&2; exit 2; }
    CASES+=(1011); LABELS+=("pi05-taskfixed-rep${REPEAT_ID}-W${w}"); PLANS+=("$PLAN_DIR/D0-W${w}.npz")
  done
fi
(( ${#CASES[@]} )) || { echo 'Empty experiment selection' >&2; exit 2; }
declare -A SEEN=()
for i in "${!CASES[@]}"; do
  item="${CASES[$i]}"
  name="${LABELS[$i]}-base-vlm2-P${item:0:1}-D${item:1:1}-E${item:2:1}-T${item:3:1}"
  [[ -z "${SEEN[$name]:-}" ]] || { echo "Duplicate run: $name" >&2; exit 2; }
  SEEN[$name]=1
  [[ ! -e "$OUTPUT_ROOT/$name" ]] || { echo "Output already exists: $OUTPUT_ROOT/$name" >&2; exit 1; }
  RUN_DIRS+=("$OUTPUT_ROOT/$name")
done
python -m lerobot.scripts.pdet_experiment check-inits "$PROMPT_INIT_DIR"
if [[ "$MODE" == order ]]; then
  python -m lerobot.scripts.pdet_followup prepare --dataset-root "$DATASET_ROOT" \
    --output "$PLAN_DIR" --within-seeds "${REQUESTED[@]}"
else
  # Check trace metadata before the first expensive model construction.
  python - "$DATASET_ROOT" <<'PY'
import sys
from lerobot.utils.pdet_order import read_frame_metadata
frames = read_frame_metadata(sys.argv[1])
print(f"Coverage metadata OK: {len(frames['index'])} frames")
PY
fi
run_one() {
  local i="$1" dry="$2" item="${CASES[$1]}"
  local init_path
  init_path=$(python -m lerobot.scripts.export_pi05_prompt_init resolve-libero-init "$PROMPT_INIT_DIR" --seed "${item:0:1}")
  local extra=()
  if [[ -n "${PLANS[$i]}" ]]; then extra+=(--pdet.order_plan="${PLANS[$i]}"); fi
  RUN_NAME="$(basename -- "${RUN_DIRS[$i]}")" \
  PROMPT_INIT_PATH="$init_path" \
  VLM_PROMPT_TOKENS=2 DRY_RUN="$dry" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
    --policy.compile_model=false \
    --pdet.enabled=true --pdet.prompt_seed="${item:0:1}" --pdet.data_seed="${item:1:1}" \
    --pdet.noise_seed="${item:2:1}" --pdet.time_seed="${item:3:1}" \
    --pdet.audit_steps=3000 --pdet.diagnostics=true \
    --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
    --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
    --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
    --log_freq=10 "${extra[@]}" "${WANDB_ARGS[@]}"
}
WANDB_ARGS=("$@")
for i in "${!CASES[@]}"; do run_one "$i" true >/dev/null; done
echo "All ${#CASES[@]} preflights passed; mode=$MODE GPUs=$GPU_IDS port=$MAIN_PROCESS_PORT"
for i in "${!CASES[@]}"; do
  echo "Starting $(basename -- "${RUN_DIRS[$i]}")"
  run_one "$i" "$DRY_RUN"
  if [[ "$DRY_RUN" == false ]]; then
    python -m lerobot.scripts.pdet_followup summarize "${RUN_DIRS[$i]}"
    if [[ "$MODE" == repeats ]]; then
      item="${CASES[$i]}"
      original="$OUTPUT_ROOT/pi05-pdet-base-vlm2-P${item:0:1}-D${item:1:1}-E${item:2:1}-T${item:3:1}"
      if [[ -f "$original/pdet/manifest-rank0.json" ]]; then
        python -m lerobot.scripts.pdet_followup compare-inputs "$original" "${RUN_DIRS[$i]}" \
          | tee "${RUN_DIRS[$i]}/pdet_analysis/original_input_comparison.json"
      fi
    fi
  fi
done
if [[ "$DRY_RUN" == false && ${#CASES[@]} -gt 1 ]]; then
  if [[ "$MODE" == order ]]; then
    python -m lerobot.scripts.pdet_followup audit-order "${RUN_DIRS[@]}"
  else
    python -m lerobot.scripts.pdet_experiment audit "${RUN_DIRS[@]}"
  fi
fi
