#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_pi05_curriculum_ablation.sh [--wandb.*=...]
One invocation runs four rounds, with a barrier after each pair:
  GPU_IDS_A=0,1                 GPU_IDS_B=2,3
1 B E2H easy30/hard70 joint     C random easy30/hard70 joint
2 D H2E easy50/hard50 joint     F random easy50/hard50 joint
3 E E2H easy50/hard50 joint     A H2E easy30/hard70 joint (repeat)
4 G H2E easy30/hard70 expert    H H2E easy30/hard70 prompt-only
G preserves the exact imported P1 tokens but freezes them. All runs start from
pi05_libero_base, not the preceding run. Fixed P1/D0/E1/T1/Bocc, VLM2, S0,
2 GPUs x batch32, bf16, no compile. Default FLOW_STEPS=6000, REPEAT_ID=2.
Every run saves only its final full checkpoint. No initial/intermediate models.
All-run CPU preflight; skip only verified completed outputs. A partial output is
never overwritten/resumed/deleted. A training failure stops the suite and both lanes.
Controls: GPU_IDS_A/B, PORT_A=29588, PORT_B=29589, FLOW_STEPS, REPEAT_ID,
OUTPUT_ROOT, LOG_ROOT, SOURCE_PAIR_DIR, SCORES_PATH, REFERENCE_RUN, DATASET_ROOT,
TOKENIZER_PATH, CURRICULUM_PLAN_ROOT, DRY_RUN=true (CPU preparation only).
Status: LOG_ROOT/curriculum-ablation-r2-steps6000/{runner.log,status.json,audit.json}.
Use nohup or tmux to keep the one invocation running after disconnecting.
EOF
  exit 0
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m lerobot.scripts.pi05_curriculum_ablation run "$@"
