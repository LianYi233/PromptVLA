#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# The Python entry point keeps --dry-run independent of plans, models, and GPUs.
exec python -u -m lerobot.scripts.pi05_task7_train_eval "$@"
