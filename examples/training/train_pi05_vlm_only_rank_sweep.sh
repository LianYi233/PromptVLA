#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_vlm_only_rank_sweep.sh [SEED=1] [extra args...]
Sequential base-model runs: VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32", seed 1.
Two GPUs (GPU_IDS=0,1), batch 32 per GPU; 3000 updates; only final full checkpoint.
Run names: pi05-base-vlm{1,2,4,8,16,32}-seed1. Existing runs are never overwritten.
Same inputs/output roots and W&B settings as train_pi05_vlm_only_token_sweep.sh.
PROMPT_INIT_PATH imports a portable step-zero prompt; select its single token count.
PROMPT_EXPORT_PATH exports a fresh initial prompt and exits BEFORE optimizer/probes/updates.
PROMPT_EXPORT_REFERENCE optionally checks an earlier swap's pre-import hash in prompt_init.json.
The positional SEED still controls data order, flow noise/time and other training randomness.

Fixed-probe diagnostics: step 0, every 250 updates, and the final step.
Defaults: RANK_FREQ=250, RANK_SAMPLES=16, RANK_OUTPUT_PROBES=4, RANK_SEED=2026,
RANK_TIMESTEPS='[0.2,0.5,0.8]'. This estimates the output-projected velocity
Jacobian rank (192-row ceiling by default), NOT the full Jacobian's rank.
Writes prompt_rank/{metrics.csv,metrics.jsonl,manifest.json,fixed_probe.pt,step_*.npz}
within each run and eval/prompt_rank/* curves to W&B. NPZ files contain small prompt
snapshots and Gram spectra, not full model checkpoints. No torch.compile.

GPU_IDS=0,1 BATCH_SIZE=32 bash examples/training/train_pi05_vlm_only_rank_sweep.sh 1
DRY_RUN=true bash examples/training/train_pi05_vlm_only_rank_sweep.sh 1
EOF
  exit 0
fi
SEED="${1:-1}"
if (( $# )); then shift; fi
export VLM_PROMPT_TOKEN_COUNTS="${VLM_PROMPT_TOKEN_COUNTS-1 2 4 8 16 32}"
RANK_FREQ="${RANK_FREQ:-250}"
RANK_SAMPLES="${RANK_SAMPLES:-16}"
RANK_OUTPUT_PROBES="${RANK_OUTPUT_PROBES:-4}"
RANK_SEED="${RANK_SEED:-2026}"
RANK_TIMESTEPS="${RANK_TIMESTEPS:-[0.2,0.5,0.8]}"
for key in RANK_FREQ RANK_SAMPLES RANK_OUTPUT_PROBES; do
  if [[ ! "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "$key must be positive" >&2; exit 2; fi
done
if [[ ! "$RANK_SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid RANK_SEED" >&2; exit 2; fi
python - "$RANK_TIMESTEPS" <<'PY'
import json
import math
import sys
try:
    times = json.loads(sys.argv[1])
    if not isinstance(times, list) or not times or any(
        type(t) not in (int, float) or not math.isfinite(t) or not 0 < t < 1 for t in times
    ) or len(set(times)) != len(times):
        raise ValueError
except (ValueError, TypeError):
    sys.exit('RANK_TIMESTEPS must be a JSON list of distinct numbers strictly between 0 and 1')
PY
for arg in "$@"; do
  case "$arg" in
    --prompt_rank.*|--policy.compile_model=*) echo "Use RANK_* controls; this sweep requires no compile" >&2; exit 2 ;;
  esac
done
exec bash "$SCRIPT_DIR/train_pi05_vlm_only_token_sweep.sh" base "$SEED" \
  --prompt_rank.freq="$RANK_FREQ" --prompt_rank.samples="$RANK_SAMPLES" \
  --prompt_rank.output_probes="$RANK_OUTPUT_PROBES" --prompt_rank.seed="$RANK_SEED" \
  --prompt_rank.timesteps="$RANK_TIMESTEPS" "$@"
