#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_two_frame.sh E2H|H2E [--wandb.*=...]
Exactly TWO observation/chunk starts. Prepare once BEFORE the two concurrent jobs:
  PYTHONPATH=src python -m lerobot.scripts.pi05_two_frame prepare --task-index 0 --steps 6000
Default PAIR_DIR=/data1/wyn/prompt-work/twoframe-r1; FLOW_STEPS=6000 (must match plan).
Equal 3000-update phases; each batch repeats the current frame with independent E/T.
Both orders share P1, the same frame/occurrence/noise bank and full action expert + VLM2.
Fixed: 2 GPUs x batch32, E1/T1/Bocc, global0, bf16, no compile, final-only checkpoint.
Defaults: GPU_IDS=0,1 MAIN_PROCESS_PORT=29586 REPEAT_ID=1 DRY_RUN=false.
Fixed two-frame losses are measured before training, at the switch/final, and every250 steps.
Controls: PAIR_DIR, SOURCE_PAIR_DIR, OUTPUT_ROOT, LOG_ROOT, DATASET_ROOT, TOKENIZER_PATH.
This is a sequential two-frame learning/forgetting diagnostic, not a dataset-wide curriculum claim.
EOF
  exit 0
fi
DIRECTION="${1:-}"
case "$DIRECTION" in E2H|H2E) shift ;; *) echo 'Choose E2H or H2E; see --help' >&2; exit 2 ;; esac
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; this experiment controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export GPU_IDS="${GPU_IDS:-0,1}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29586}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS="${FLOW_STEPS:-6000}" SAVE_ONLY_FINAL=true DTYPE=bfloat16
[[ "$FLOW_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid FLOW_STEPS' >&2; exit 2; }
if [[ "${TRAIN_ACTION_EXPERT:-true}" != true ]]; then echo 'Requires TRAIN_ACTION_EXPERT=true' >&2; exit 2; fi
export TRAIN_ACTION_EXPERT=true
REPEAT_ID="${REPEAT_ID:-1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
case "${DRY_RUN:-false}" in true|false) ;; *) echo 'Invalid DRY_RUN' >&2; exit 2 ;; esac
PAIR_DIR="${PAIR_DIR:-/data1/wyn/prompt-work/twoframe-r1}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
[[ -f "$PAIR_DIR/pair.json" ]] || {
  echo "Missing $PAIR_DIR/pair.json; run python -m lerobot.scripts.pi05_two_frame prepare first (see --help)" >&2; exit 1;
}
PREFIX=$(python -m lerobot.scripts.pi05_two_frame prefix "$PAIR_DIR" "$DIRECTION" --repeat "$REPEAT_ID" --steps "$FLOW_STEPS")
export RUN_NAME="${PREFIX}-base-vlm2-P1-D0-E1-T1"
[[ ! -e "$OUTPUT_ROOT/$RUN_NAME" ]] || { echo "Output already exists: $OUTPUT_ROOT/$RUN_NAME" >&2; exit 1; }
python - "$PAIR_DIR/pair.json" "$SOURCE_PAIR_DIR" "$DATASET_ROOT" <<'PY'
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if Path(record["source_pair"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("SOURCE_PAIR_DIR differs from prepared two-frame pair")
source = json.loads((Path(sys.argv[2]) / "pair.json").read_text())
if Path(source["dataset_root"]).resolve() != Path(sys.argv[3]).resolve():
    raise SystemExit("DATASET_ROOT differs from prepared source")
PY
SWITCH_STEP=$((FLOW_STEPS / 2))
PRETRAINED_PATH="$SOURCE_PAIR_DIR/libero" VLM_PROMPT_TOKENS=2 \
PROMPT_INIT_PATH="$SOURCE_PAIR_DIR/P1.safetensors" \
  bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
  --policy.compile_model=false \
  --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
  --pdet.noise_seed=1 --pdet.time_seed=1 --pdet.flow_binding=occurrence \
  --pdet.order_plan="$PAIR_DIR/$DIRECTION.npz" --pdet.audit_steps="$FLOW_STEPS" --pdet.diagnostics=true \
  --prompt_rank.freq=250 --prompt_rank.samples=2 --prompt_rank.output_probes=4 \
  --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
  --prompt_rank.extra_steps="[50,100,$SWITCH_STEP]" --prompt_rank.fixed_flow_loss=true \
  --log_freq=10 "$@"
if [[ "${DRY_RUN:-false}" == false ]]; then
  cp -- "$PAIR_DIR/pair.json" "$OUTPUT_ROOT/$RUN_NAME/two_frame_pair.json"
  python -m lerobot.scripts.pi05_two_frame audit "$PAIR_DIR" "$OUTPUT_ROOT/$RUN_NAME"
fi
