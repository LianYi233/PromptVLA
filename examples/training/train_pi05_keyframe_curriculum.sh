#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_keyframe_curriculum.sh a|b baseline|weight|curriculum [--wandb.*=...]
Two independent two-GPU queues; prepare the archive ONCE before either queue.
             a (GPU0,1 / 29571)        b (GPU2,3 / 29572)
baseline:    B0                       B1
weight:      K2:1, R2:0               K2:0, R2:1
curriculum:  Kearly:1, Klate:0         Kearly:0, Klate:1
Bank b means E=b,T=b jointly. Run baseline first; review before weight/curriculum.
Fixed: base, VLM2, P1/D0/W0/Bocc, seed0, 3000 steps, batch32/GPU, compile=false,
original LR recipe, fixed probes, only final full checkpoint003000, W&B online.
CELLS="K2:1 R2:0" selects only listed cells in THIS queue/stage after interruption.
Controls: EXPERIMENT_ID, KEYFRAME_DIR, GPU_IDS, MAIN_PROCESS_PORT, OUTPUT_ROOT,
LOG_ROOT, DATASET_ROOT, BASE_PRETRAINED_PATH, TOKENIZER_PATH, PROMPT_INIT_DIR,
DRY_RUN=true. Existing outputs are never overwritten or silently skipped.
No new sampler, no new initialization, no extra 500-step warmup, no GPU scoring.
EOF
  exit 0
fi
QUEUE="${1:-}"; STAGE="${2:-}"
(( $# >= 2 )) || { echo 'Expected a|b baseline|weight|curriculum; see --help' >&2; exit 2; }
shift 2
case "$QUEUE" in a) DEFAULT_GPUS=0,1; DEFAULT_PORT=29571 ;; b) DEFAULT_GPUS=2,3; DEFAULT_PORT=29572 ;; *) echo 'Invalid queue' >&2; exit 2 ;; esac
case "$QUEUE:$STAGE" in
  a:baseline) DEFAULT_CELLS='B:0' ;; b:baseline) DEFAULT_CELLS='B:1' ;;
  a:weight) DEFAULT_CELLS='K2:1 R2:0' ;; b:weight) DEFAULT_CELLS='K2:0 R2:1' ;;
  a:curriculum) DEFAULT_CELLS='Kearly:1 Klate:0' ;; b:curriculum) DEFAULT_CELLS='Kearly:0 Klate:1' ;;
  *) echo 'Invalid stage' >&2; exit 2 ;;
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
KEYFRAME_DIR="${KEYFRAME_DIR:-/data1/wyn/prompt-work/keyframe-r${EXPERIMENT_ID}}"
[[ "$EXPERIMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EXPERIMENT_ID' >&2; exit 2; }
case "$DRY_RUN" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
read -r -a SELECTED <<< "${CELLS-$DEFAULT_CELLS}"
(( ${#SELECTED[@]} )) || { echo 'Empty CELLS selection' >&2; exit 2; }
declare -A SEEN=()
RUN_DIRS=()
for cell in "${SELECTED[@]}"; do
  [[ " $DEFAULT_CELLS " == *" $cell "* && -z "${SEEN[$cell]:-}" ]] || { echo "Invalid/duplicate cell for this queue/stage: $cell" >&2; exit 2; }
  SEEN[$cell]=1
  mode="${cell%:*}"; bank="${cell#*:}"
  name="pi05-key-r${EXPERIMENT_ID}-${mode}-W0-Bocc-base-vlm2-P1-D0-E${bank}-T${bank}"
  [[ ! -e "$OUTPUT_ROOT/$name" ]] || { echo "Output already exists: $OUTPUT_ROOT/$name" >&2; exit 1; }
  RUN_DIRS+=("$OUTPUT_ROOT/$name")
done
python -m lerobot.scripts.keyframe_experiment check "$KEYFRAME_DIR" \
  --dataset-root "$DATASET_ROOT" --source "${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}"
python - "$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed1.safetensors" <<'PY'
import sys
from lerobot.utils.prompt_initialization import read_initial_prompt
_, metadata = read_initial_prompt(sys.argv[1], 2)
if metadata["source_training_seed"] != 1 or metadata.get("source_prompt_init_path"):
    raise SystemExit("P1 must be an original, unswapped seed-1 initialization")
print("P1 initialization provenance OK")
PY
WANDB_ARGS=("$@")
run_one() {
  local i="$1" dry="$2" cell="${SELECTED[$1]}"
  local mode="${cell%:*}" bank="${cell#*:}"
  RUN_NAME="$(basename -- "${RUN_DIRS[$i]}")" VLM_PROMPT_TOKENS=2 DRY_RUN="$dry" \
  PROMPT_INIT_PATH="$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed1.safetensors" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
    --policy.compile_model=false \
    --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
    --pdet.noise_seed="$bank" --pdet.time_seed="$bank" \
    --pdet.order_plan="$KEYFRAME_DIR/D0-W0.npz" --pdet.flow_binding=occurrence \
    --pdet.audit_steps=3000 --pdet.diagnostics=true \
    --keyframe_loss.mode="$mode" --keyframe_loss.archive="$KEYFRAME_DIR/archive.npz" \
    --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
    --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
    --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
    --log_freq=10 "${WANDB_ARGS[@]}"
}
for i in "${!RUN_DIRS[@]}"; do run_one "$i" true >/dev/null; done
echo "All ${#RUN_DIRS[@]} preflights passed; queue=$QUEUE stage=$STAGE GPUs=$GPU_IDS port=$MAIN_PROCESS_PORT"
for i in "${!RUN_DIRS[@]}"; do
  echo "Starting $(basename -- "${RUN_DIRS[$i]}")"
  run_one "$i" "$DRY_RUN"
done
if [[ "$DRY_RUN" == false ]]; then
  python -m lerobot.scripts.keyframe_experiment audit --stage "$STAGE" --allow-partial "${RUN_DIRS[@]}"
fi
