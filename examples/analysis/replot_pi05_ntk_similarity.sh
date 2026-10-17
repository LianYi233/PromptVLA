#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash $0 RUN_DIR [--scope=backbone|prompts|both] [--full-matrix] [extra arguments...]" >&2
  exit 2
fi
RUN_DIR="${1%/}"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m lerobot.scripts.plot_pi05_ntk_similarity \
  "${NTK_RESULTS:-${RUN_DIR}/ntk_stages/results.json}" "$@"
