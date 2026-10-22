#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_frameskip_curriculum.sh a|b [--wandb.*=...]
Prepare scores/plans once first (see examples/analysis/README_frameskip_curriculum.md).
a: GPUs0,1 port29561, cells U0 R1 S0 C1; b: GPUs2,3 port29562, U1 R0 S1 C0.
Cell letter: U uniform, R random pool, S static scored pool, C scored curriculum.
Cell digit: joint E/T bank. All P1 D0, 2 tokens, occurrence binding, 3000 updates,
500 identical full-frame warmup updates, global batch64, final full checkpoint only.
S/C use identical finite sample-use multisets AND occurrence-bound flow draws.
CELLS="S0 C1" can explicitly select remaining unstarted cells. Existing outputs
always fail; EXPERIMENT_ID changes names only, not stochastic inputs.
Controls: GPU_IDS, MAIN_PROCESS_PORT, EXPERIMENT_ID=1, SCORE_DIR, PLAN_DIR,
OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT, BASE_PRETRAINED_PATH, TOKENIZER_PATH,
PROMPT_INIT_DIR, DRY_RUN=true. No implicit plan regeneration while training.
EOF
  exit 0
fi
QUEUE="${1:-}"; shift || true
case "$QUEUE" in
  a) DEFAULT_GPUS=0,1; DEFAULT_PORT=29561; DEFAULT_CELLS='U0 R1 S0 C1' ;;
  b) DEFAULT_GPUS=2,3; DEFAULT_PORT=29562; DEFAULT_CELLS='U1 R0 S1 C0' ;;
  *) echo 'Expected a or b; see --help' >&2; exit 2 ;;
esac
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  [[ -z "${!key:-}" ]] || { echo "Unset $key; this experiment controls it" >&2; exit 2; }
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export BASE_PRETRAINED_PATH="${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}"
export GPU_IDS="${GPU_IDS:-$DEFAULT_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$DEFAULT_PORT}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
PROMPT_INIT_DIR="${PROMPT_INIT_DIR:-/data1/wyn/prompt-work}"
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"; DRY_RUN="${DRY_RUN:-false}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
case "$DRY_RUN" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
SCORE_DIR="${SCORE_DIR:-/data1/wyn/prompt-work/frameskip-r${EXPERIMENT_ID}}"
PLAN_DIR="${PLAN_DIR:-$SCORE_DIR/plans}"
read -r -a REQUESTED <<< "${CELLS-$DEFAULT_CELLS}"
(( ${#REQUESTED[@]} )) || { echo 'Empty CELLS' >&2; exit 2; }
declare -A SEEN=()
RUN_DIRS=()
for cell in "${REQUESTED[@]}"; do
  [[ "$cell" =~ ^[URSC][01]$ && -z "${SEEN[$cell]:-}" ]] || { echo "Invalid/duplicate cell: $cell" >&2; exit 2; }
  SEEN[$cell]=1
  b="${cell:1:1}"
  name="pi05-fskip-r${EXPERIMENT_ID}-${cell:0:1}-Bocc-base-vlm2-P1-D0-E${b}-T${b}"
  [[ ! -e "$OUTPUT_ROOT/$name" ]] || { echo "Output already exists: $OUTPUT_ROOT/$name; select unstarted CELLS" >&2; exit 1; }
  RUN_DIRS+=("$OUTPUT_ROOT/$name")
done
python -m lerobot.scripts.frameskip_experiment audit-plans "$PLAN_DIR" --scores "$SCORE_DIR/scores.npz"
python -m lerobot.scripts.frameskip_experiment check-inputs --scores "$SCORE_DIR/scores.npz" \
  --dataset-root "$DATASET_ROOT" --source "$BASE_PRETRAINED_PATH"
python - "$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed1.safetensors" <<'PY'
import sys
from lerobot.utils.prompt_initialization import read_initial_prompt
_, metadata = read_initial_prompt(sys.argv[1], 2)
if metadata['source_training_seed'] != 1 or metadata.get('source_prompt_init_path'):
    raise SystemExit('P1 must be the original unswapped seed-1 initialization')
PY
WANDB_ARGS=("$@")
run_one() {
  local i="$1" dry="$2" cell="${REQUESTED[$1]}"
  local b="${cell:1:1}" schedule="${cell:0:1}"
  RUN_NAME="$(basename -- "${RUN_DIRS[$i]}")" VLM_PROMPT_TOKENS=2 DRY_RUN="$dry" \
  PROMPT_INIT_PATH="$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed1.safetensors" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
    --policy.compile_model=false \
    --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
    --pdet.noise_seed="$b" --pdet.time_seed="$b" \
    --pdet.order_plan="$PLAN_DIR/$schedule.npz" --pdet.flow_binding=occurrence \
    --pdet.audit_steps=3000 --pdet.diagnostics=true \
    --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
    --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
    --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
    --log_freq=10 "${WANDB_ARGS[@]}"
}
for i in "${!REQUESTED[@]}"; do run_one "$i" true >/dev/null; done
echo "All ${#REQUESTED[@]} preflights passed; queue=$QUEUE GPUs=$GPU_IDS port=$MAIN_PROCESS_PORT"
for i in "${!REQUESTED[@]}"; do
  echo "Starting $(basename -- "${RUN_DIRS[$i]}")"
  run_one "$i" "$DRY_RUN"
  if [[ "$DRY_RUN" == false ]]; then
    python -m lerobot.scripts.pdet_followup summarize "${RUN_DIRS[$i]}"
  fi
done
if [[ "$DRY_RUN" == false ]]; then
  python -m lerobot.scripts.frameskip_experiment audit-runs --allow-partial "${RUN_DIRS[@]}"
fi
