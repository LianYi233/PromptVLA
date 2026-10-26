#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_source_compare.sh libero|generic [--wandb.*=...]
libero = pi05_libero_base repeat (GPU0,1 / port29581).
generic = pi05_base weights + the SAME LIBERO config/processors (GPU2,3 / port29582).
Prepare once with python -m lerobot.scripts.pi05_source_compare prepare.
Fixed: saved P1 tensor, original D0/W0/Bocc plan, E1/T1, global seed0, VLM2,
3000 steps, batch32/GPU, 2 GPUs/run, bf16, no compile, original unweighted loss.
Only full checkpoint003000; original fixed probes and full PDET traces retained.
No resampling, new initialization, key-action weighting or curriculum.
Controls: REPEAT_ID=1, SOURCE_PAIR_DIR, GPU_IDS, MAIN_PROCESS_PORT, OUTPUT_ROOT,
LOG_ROOT, TOKENIZER_PATH, DATASET_ROOT, DRY_RUN=true.
The two runs differ only in original weights; all original model files stay unchanged.
EOF
  exit 0
fi
SOURCE="${1:-}"; shift || true
case "$SOURCE" in
  libero) DEFAULT_GPUS=0,1; DEFAULT_PORT=29581 ;;
  generic) DEFAULT_GPUS=2,3; DEFAULT_PORT=29582 ;;
  *) echo 'Expected libero or generic; see --help' >&2; exit 2 ;;
esac
for arg in "$@"; do
  case "$arg" in --wandb.*=*) ;; *) echo "Unsupported override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH BASE_PRETRAINED_PATH RUN_NAME RUN_PREFIX VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE PDET_CASES; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; source comparison controls it" >&2; exit 2; fi
done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export GPU_IDS="${GPU_IDS:-$DEFAULT_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$DEFAULT_PORT}"
export BATCH_SIZE=32 NUM_PROCESSES=2 FLOW_STEPS=3000 SAVE_ONLY_FINAL=true DTYPE=bfloat16
REPEAT_ID="${REPEAT_ID:-1}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-/data1/wyn/prompt-work/source-pair-r1}"
[[ "$REPEAT_ID" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid REPEAT_ID' >&2; exit 2; }
export RUN_NAME="pi05-source-r${REPEAT_ID}-${SOURCE}-W0-Bocc-base-vlm2-P1-D0-E1-T1"
[[ ! -e "$OUTPUT_ROOT/$RUN_NAME" ]] || { echo "Output already exists: $OUTPUT_ROOT/$RUN_NAME" >&2; exit 1; }
python -m lerobot.scripts.pi05_source_compare check "$SOURCE_PAIR_DIR"
python - "$SOURCE_PAIR_DIR/pair.json" "$DATASET_ROOT" <<'PY'
import json, sys
from pathlib import Path
if Path(json.loads(Path(sys.argv[1]).read_text())["dataset_root"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("DATASET_ROOT differs from the prepared comparison")
PY
PRETRAINED_PATH="$SOURCE_PAIR_DIR/$SOURCE" VLM_PROMPT_TOKENS=2 \
PROMPT_INIT_PATH="$SOURCE_PAIR_DIR/P1.safetensors" \
  bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
  --policy.compile_model=false \
  --pdet.enabled=true --pdet.prompt_seed=1 --pdet.data_seed=0 \
  --pdet.noise_seed=1 --pdet.time_seed=1 \
  --pdet.order_plan="$SOURCE_PAIR_DIR/D0-W0.npz" --pdet.flow_binding=occurrence \
  --pdet.audit_steps=3000 --pdet.diagnostics=true \
  --prompt_rank.freq=250 --prompt_rank.samples=16 --prompt_rank.output_probes=4 \
  --prompt_rank.seed=2026 --prompt_rank.timesteps='[0.2,0.5,0.8]' \
  --prompt_rank.extra_steps='[50,100]' --prompt_rank.fixed_flow_loss=true \
  --log_freq=10 "$@"
if [[ "${DRY_RUN:-false}" == false ]]; then
  cp -- "$SOURCE_PAIR_DIR/pair.json" "$OUTPUT_ROOT/$RUN_NAME/source_pair.json"
fi
