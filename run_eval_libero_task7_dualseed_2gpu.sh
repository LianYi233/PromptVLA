#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1}"
# Fixed scene RNG and separately fixed policy RNG; calibration and heldout
# policy seeds are disjoint. The old task7 shift protocol remains untouched.
exec python -u -m lerobot.scripts.pi05_task7_dualseed_eval "$@"
