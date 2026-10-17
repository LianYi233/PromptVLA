#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_only_pdet_sweep.sh [--wandb.*=...]
Base model, VLM-only token=2. P=imported prompt, D=data order, E=Gaussian noise, T=Beta time.
Default PDET_CASES="1111 0000 0111 1011 1101 1110 1000" (digits in P,D,E,T order).
Seven sequential runs, each using GPU_IDS=0,1, BATCH_SIZE=32 PER GPU, 3000 updates.
Only checkpoint 003000 is saved; no torch.compile. Other RNG uses fixed global seed 0.
Names: pi05-pdet-base-vlm2-P1-D1-E1-T1 (RUN_PREFIX defaults to pi05-pdet).
This independent-stream protocol does NOT replay historical global-seed noise/time draws.

PROMPT_INIT_DIR defaults to /data1/wyn/prompt-work and must contain:
  pi05-base-vlm2-init-seed0.safetensors and pi05-base-vlm2-init-seed1.safetensors
Both are checked against their provenance and must have different tensor hashes.
Inputs default to /home/wyn/{models,datasets}; checkpoints default to
/data1/wyn/chkpt/2601-lerobot/PromptVLA; logs to /data1/wyn/chkpt/2601-lerobot/logs.
No overwrites/skipping: after partial completion choose remaining PDET_CASES explicitly.
DRY_RUN=true checks all inputs/outputs and prints commands without starting training.

Rank/NTK probes: RANK_FREQ=250 (0 disables), RANK_SAMPLES=16, RANK_OUTPUT_PROBES=4,
RANK_SEED=2026, RANK_TIMESTEPS='[0.2,0.5,0.8]'. Same probes across all runs.
Every rank writes pdet/manifest-rankN.json and hashes/sample IDs for the first AUDIT_STEPS=8.
All selected runs pass preflight before training starts; stop on first failure.
EOF
  exit 0
fi
for arg in "$@"; do
  case "$arg" in --wandb.*=*|--log_freq=*) ;; *) echo "Unsupported PDET override: $arg" >&2; exit 2 ;; esac
done
for key in PRETRAINED_PATH RUN_NAME VLM_PROMPT_TOKENS PROMPT_INIT_PATH PROMPT_EXPORT_PATH PROMPT_EXPORT_REFERENCE; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key; PDET sweep controls it" >&2; exit 2; fi
done
read -r -a CASES <<< "${PDET_CASES-1111 0000 0111 1011 1101 1110 1000}"
if (( ${#CASES[@]} == 0 )); then echo "PDET_CASES cannot be empty" >&2; exit 2; fi
declare -A SEEN=()
for item in "${CASES[@]}"; do
  if [[ ! "$item" =~ ^[01]{4}$ || -n "${SEEN[$item]:-}" ]]; then
    echo "Invalid/duplicate case: $item (use four binary digits in P,D,E,T order)" >&2; exit 2
  fi
  SEEN[$item]=1
done
DRY_RUN="${DRY_RUN:-false}"
case "$DRY_RUN" in true|false) ;; *) echo "Invalid DRY_RUN" >&2; exit 2 ;; esac
RUN_PREFIX="${RUN_PREFIX:-pi05-pdet}"
if [[ ! "$RUN_PREFIX" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo "Invalid RUN_PREFIX" >&2; exit 2; fi
PROMPT_INIT_DIR="${PROMPT_INIT_DIR:-/data1/wyn/prompt-work}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1}" BATCH_SIZE="${BATCH_SIZE:-32}"
export FLOW_STEPS=3000 SAVE_ONLY_FINAL=true
export BASE_PRETRAINED_PATH="${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wyn/datasets/libero}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export LOG_ROOT="${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}"
EXTRA_ARGS=(--policy.compile_model=false
  --prompt_rank.freq="${RANK_FREQ:-250}" --prompt_rank.samples="${RANK_SAMPLES:-16}"
  --prompt_rank.output_probes="${RANK_OUTPUT_PROBES:-4}" --prompt_rank.seed="${RANK_SEED:-2026}"
  --prompt_rank.timesteps="${RANK_TIMESTEPS:-[0.2,0.5,0.8]}" "$@")
python -m lerobot.scripts.pdet_experiment check-inits "$PROMPT_INIT_DIR"
# Validate diagnostics before launching any expensive model.
python - "${RANK_FREQ:-250}" "${RANK_SAMPLES:-16}" "${RANK_OUTPUT_PROBES:-4}" \
  "${RANK_SEED:-2026}" "${RANK_TIMESTEPS:-[0.2,0.5,0.8]}" "${AUDIT_STEPS:-8}" <<'PY'
import json
import sys
from lerobot.configs.pdet import PDETConfig
from lerobot.configs.prompt_rank import PromptRankConfig
freq, samples, probes, seed, times, audit = sys.argv[1:]
PromptRankConfig(int(freq), int(samples), int(probes), json.loads(times), int(seed)).validate()
PDETConfig(audit_steps=int(audit)).validate()
PY
run_one() {
  local item="$1" dry="$2"
  local p="${item:0:1}" d="${item:1:1}" e="${item:2:1}" t="${item:3:1}"
  RUN_NAME="${RUN_PREFIX}-base-vlm2-P${p}-D${d}-E${e}-T${t}" \
  PROMPT_INIT_PATH="$PROMPT_INIT_DIR/pi05-base-vlm2-init-seed${p}.safetensors" \
  VLM_PROMPT_TOKENS=2 DRY_RUN="$dry" \
    bash "$SCRIPT_DIR/train_pi05_vlm_only.sh" base 0 \
    --pdet.enabled=true --pdet.prompt_seed="$p" --pdet.data_seed="$d" \
    --pdet.noise_seed="$e" --pdet.time_seed="$t" --pdet.audit_steps="${AUDIT_STEPS:-8}" \
    "${EXTRA_ARGS[@]}"
}
echo "Checking ${#CASES[@]} PDET runs: ${CASES[*]} (global seed fixed to 0)"
for item in "${CASES[@]}"; do run_one "$item" true >/dev/null; done
echo "All PDET preflights passed; output=$OUTPUT_ROOT"
for item in "${CASES[@]}"; do
  echo "Starting P${item:0:1}-D${item:1:1}-E${item:2:1}-T${item:3:1} (3000 steps, final checkpoint only)"
  run_one "$item" "$DRY_RUN"
done
