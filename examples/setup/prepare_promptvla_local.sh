#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/setup/prepare_promptvla_local.sh [assets|inits|all]
Default: all. Run inside the local server's configured lerobot environment.
assets: download the complete lerobot/libero-assets dataset to
        /data1/datasets/libero-assets, configure LIBERO and link package assets.
        Existing asset paths/configs are backed up. No GPU needed.
inits:  sequentially construct fresh pi05_libero_base VLM2 prompts for seeds 0/1;
        export to /data1/wyn/prompt-work with zero updates, no full checkpoint/W&B.
        Names: pi05_libero_base-vlm2-init-seed{0,1}.safetensors.
        This loads the full policy using GPU_IDS=0,1 and BATCH_SIZE=32 per GPU.
        New local initializations are NOT claimed identical to historical prompts.
Overrides: LIBERO_ASSETS_PATH, LIBERO_CONFIG_PATH, PROMPT_INIT_DIR, GPU_IDS,
           BATCH_SIZE, MAIN_PROCESS_PORT, INIT_SEEDS (0 and/or 1), RUN_PREFIX.
Training input/output path overrides are inherited from the VLM-only launcher.
Existing initialization files are never overwritten. After a partial export,
select the missing seed explicitly, e.g. INIT_SEEDS=1 ... inits.
EOF
  exit 0
fi
if (( $# > 1 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
MODE="${1:-all}"
case "$MODE" in assets|inits|all) ;; *) echo "Invalid mode; see --help" >&2; exit 2 ;; esac
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PROMPT_INIT_DIR="${PROMPT_INIT_DIR:-/data1/wyn/prompt-work}"
export GPU_IDS="${GPU_IDS:-0,1}" BATCH_SIZE="${BATCH_SIZE:-32}"
export RUN_PREFIX="${RUN_PREFIX:-pi05-local-init}"
run_export() {
  local seed="$1" dry="$2"
  env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
      -u PROMPT_INIT_PATH -u PROMPT_EXPORT_REFERENCE \
    VLM_PROMPT_TOKEN_COUNTS=2 DRY_RUN="$dry" \
    PROMPT_EXPORT_PATH="$PROMPT_INIT_DIR/pi05_libero_base-vlm2-init-seed${seed}.safetensors" \
    bash "$REPO_ROOT/examples/training/train_pi05_vlm_only_token_sweep.sh" base "$seed" \
      --wandb.enable=false
}
if [[ "$MODE" == inits || "$MODE" == all ]]; then
  python - "${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}" <<'PY'
import sys
from pathlib import Path
if Path(sys.argv[1]).name != "pi05_libero_base":
    raise SystemExit("This setup exports pi05_libero_base initializers; another source needs an explicit export filename")
PY
  read -r -a SEEDS <<< "${INIT_SEEDS-0 1}"
  if (( ${#SEEDS[@]} == 0 )); then echo "INIT_SEEDS cannot be empty" >&2; exit 2; fi
  declare -A SEEN=()
  for seed in "${SEEDS[@]}"; do
    if [[ ! "$seed" =~ ^[01]$ || -n "${SEEN[$seed]:-}" ]]; then
      echo "INIT_SEEDS must contain unique values from 0 and 1" >&2; exit 2
    fi
    SEEN[$seed]=1
    if [[ -e "$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed${seed}.safetensors" || \
          -L "$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed${seed}.safetensors" ]]; then
      echo 'Existing legacy initialization: use export_pi05_prompt_init rename-libero-inits; do not regenerate it.' >&2
      exit 1
    fi
    run_export "$seed" true >/dev/null
  done
  echo "Initialization export preflight passed: seeds=${SEEDS[*]}, tokens=2, GPUs=$GPU_IDS"
fi
if [[ "$MODE" == assets || "$MODE" == all ]]; then
  python -m lerobot.scripts.setup_libero_assets \
    --assets-dir "${LIBERO_ASSETS_PATH:-/data1/datasets/libero-assets}" \
    --config-dir "${LIBERO_CONFIG_PATH:-/home/wyn/.libero}"
fi
if [[ "$MODE" == inits || "$MODE" == all ]]; then
  for seed in "${SEEDS[@]}"; do
    echo "Exporting new local initialization: token=2 seed=$seed (0 updates)"
    run_export "$seed" false
    python -m lerobot.scripts.export_pi05_prompt_init check \
      "$PROMPT_INIT_DIR/pi05_libero_base-vlm2-init-seed${seed}.safetensors" --tokens 2
  done
  if [[ -f "$PROMPT_INIT_DIR/pi05_libero_base-vlm2-init-seed0.safetensors" && \
        -f "$PROMPT_INIT_DIR/pi05_libero_base-vlm2-init-seed1.safetensors" ]]; then
    python -m lerobot.scripts.pdet_experiment check-inits "$PROMPT_INIT_DIR"
  fi
fi
