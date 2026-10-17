#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash $0 RUN_DIR [--pptx-template /path/to/editable.pptx] [extra arguments...]" >&2
  exit 2
fi
RUN_DIR="${1%/}"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m lerobot.scripts.replot_pi05_ntk_panels \
  "${NTK_RESULTS:-${RUN_DIR}/ntk_stages/results.json}" "$@"
