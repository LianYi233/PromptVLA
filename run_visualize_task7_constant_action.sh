#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# CPU only. --checkpoint reads just the action projection and time projection;
# --illustrative intentionally labels deterministic UNTRAINED weights.
exec python -u -m lerobot.scripts.pi05_task7_constant_action "$@"
