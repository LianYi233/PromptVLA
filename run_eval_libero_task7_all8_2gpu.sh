#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Each existing experiment launches two GPU lanes per phase.
# Never allocate all eight checkpoints simultaneously. DRY_RUN=true prints
# all pending invocations without loading datasets or GPU model weights.
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exec python -u -m lerobot.scripts.pi05_task7_joint_eval "$@" --dry-run
fi
exec python -u -m lerobot.scripts.pi05_task7_joint_eval "$@"
