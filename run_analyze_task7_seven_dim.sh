#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# CPU-only analysis of saved full_sampling_evidence.json/full_sampling_arrays.npz.
exec python -u -m lerobot.scripts.pi05_task7_seven_dim "$@"
