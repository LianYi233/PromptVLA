#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash $0 RUN_DIR [extra plotting arguments...]" >&2
  exit 2
fi
RUN_DIR="${1%/}"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=("${NTK_RESULTS:-${RUN_DIR}/ntk_stages/results.json}")
HAS_SOURCE=false
for arg in "$@"; do
  case "$arg" in --training-log|--training-log=*|--loss-csv|--loss-csv=*) HAS_SOURCE=true ;; esac
done
if [[ "$HAS_SOURCE" == false ]]; then
  ARGS+=(--training-log="${TRAIN_LOG:-${LOG_ROOT:-/root/autodl-tmp/logs/prompt-ablation}/$(basename -- "$RUN_DIR").log}")
fi
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m lerobot.scripts.plot_pi05_ntk_with_loss "${ARGS[@]}" "$@"
