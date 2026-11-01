#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m examples.evaluation.eval_curriculum_seed_repeat "$@"
