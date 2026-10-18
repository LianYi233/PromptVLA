#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_vlm_only_pdet_sweep.sh [all|no10|SUITE]
Evaluate base VLM-only token=2 P/D/E/T runs sequentially, checkpoint 003000.
Default PDET_CASES="1111 0000 0111 1011 1101 1110 1000": existing five, then the last two.
Names: RUN_PREFIX=pi05-pdet -> pi05-pdet-base-vlm2-P1-D1-E1-T1.
GPU_ID=2 by default, leaving training GPUs 0,1 free. Evaluation seed=1000,
10 episodes/task, n_action_steps=10; P/D/E/T are training factors only.

WAIT_FOR_CHECKPOINTS=true waits for each checkpoint's completed-save marker:
checkpoints/last must point to the requested step (written AFTER save_checkpoint).
Never evaluates a model.safetensors that is still being written. Checks happen
just before each evaluation; missing later runs do not block completed earlier runs.
WAIT_TIMEOUT_SECONDS=21600 per model; WAIT_INTERVAL_SECONDS=30 (maximum 60).
Set WAIT_FOR_CHECKPOINTS=false to fail immediately on a missing completed checkpoint.
DRY_RUN=true validates existing completed models without waiting, logs or GPU jobs.

PROGRESS=auto|true|false: one in-place episode bar with elapsed time and ETA;
stdout contains only final suite/overall rates. Full output/video paths go to logs.
Evaluation failure stops the sequence. Existing per-task resume checks still apply.
Overrides: CKPT_ROOT, BASE_OUTPUT, LOG_ROOT, EVAL_LOG_ROOT, BASE_PRETRAINED_PATH,
TOKENIZER_PATH, RUN_PREFIX, PDET_CASES, CHECKPOINT_STEP, EPISODES_PER_TASK, EVAL_SEED.
Unset RUN_NAME, PRETRAINED_PATH, VLM_PROMPT_TOKENS, BASE_CKPT and BASE_MODEL_PATH.
EOF
  exit 0
fi
if (( $# > 1 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
SUITES="${1:-all}"
case "$SUITES" in
  all) SUITE_TAG=libero-all4; TASKS=libero_spatial,libero_object,libero_goal,libero_10 ;;
  no10) SUITE_TAG=libero-no10; TASKS=libero_spatial,libero_object,libero_goal ;;
  libero_spatial|libero_object|libero_goal|libero_10) SUITE_TAG="$SUITES"; TASKS="$SUITES" ;;
  *) echo "Invalid suite; see --help" >&2; exit 2 ;;
esac
for key in RUN_NAME PRETRAINED_PATH VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key for this sweep; see --help" >&2; exit 2; fi
done
read -r -a CASES <<< "${PDET_CASES-1111 0000 0111 1011 1101 1110 1000}"
if (( ${#CASES[@]} == 0 )); then echo "PDET_CASES cannot be empty" >&2; exit 2; fi
declare -A SEEN=()
for item in "${CASES[@]}"; do
  if [[ ! "$item" =~ ^[01]{4}$ || -n "${SEEN[$item]:-}" ]]; then
    echo "Invalid/duplicate PDET case: $item" >&2; exit 2
  fi
  SEEN[$item]=1
done
RUN_PREFIX="${RUN_PREFIX:-pi05-pdet}"
if [[ ! "$RUN_PREFIX" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo "Invalid RUN_PREFIX" >&2; exit 2; fi
DRY_RUN="${DRY_RUN:-false}"
WAIT_FOR_CHECKPOINTS="${WAIT_FOR_CHECKPOINTS:-true}"
for key in DRY_RUN WAIT_FOR_CHECKPOINTS; do
  case "${!key}" in true|false) ;; *) echo "Invalid $key" >&2; exit 2 ;; esac
done
PROGRESS="${PROGRESS:-auto}"
case "$PROGRESS" in auto|true|false) ;; *) echo "Invalid PROGRESS" >&2; exit 2 ;; esac
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-30}"
export GPU_ID="${GPU_ID:-2}" CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
for key in CHECKPOINT_STEP EPISODES_PER_TASK WAIT_TIMEOUT_SECONDS WAIT_INTERVAL_SECONDS; do
  if [[ ! "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid $key" >&2; exit 2; fi
done
for key in GPU_ID EVAL_SEED; do
  if [[ ! "${!key}" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid $key" >&2; exit 2; fi
done
if (( WAIT_INTERVAL_SECONDS > 60 )); then echo "WAIT_INTERVAL_SECONDS must be <= 60" >&2; exit 2; fi
export BASE_PRETRAINED_PATH="${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export BASE_OUTPUT="${BASE_OUTPUT:-$CKPT_ROOT/eval}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}/eval}"
export TQDM_DISABLE=1
STEP_PADDED=$(printf '%06d' "$CHECKPOINT_STEP")
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
unset LIBERO_PROGRESS_FD LIBERO_PROGRESS_LABEL
if [[ "$DRY_RUN" != true && ( "$PROGRESS" == true || ( "$PROGRESS" == auto && -t 2 ) ) ]]; then
  exec 3>&2
  export LIBERO_PROGRESS_FD=3
fi
progress_status() {
  if [[ -n "${LIBERO_PROGRESS_FD:-}" ]]; then printf '\r\033[K%s' "$1" >&3; fi
}
progress_clear() {
  if [[ -n "${LIBERO_PROGRESS_FD:-}" ]]; then printf '\r\033[K' >&3; fi
}
trap progress_clear EXIT

wait_for_checkpoint() {
  local started=$SECONDS elapsed remaining delay announced=false
  # The trainer publishes this link only after weights, processors AND training state are saved.
  while [[ ! -L "$CKPT_ROOT/$NAME/checkpoints/last" || \
           ! "$CKPT_ROOT/$NAME/checkpoints/last" -ef "$CHECKPOINT" ]]; do
    if [[ "$DRY_RUN" == true || "$WAIT_FOR_CHECKPOINTS" == false ]]; then
      progress_clear
      echo "Completed checkpoint not ready: $CHECKPOINT (last link must point here)" >&2
      return 1
    fi
    elapsed=$((SECONDS - started))
    remaining=$((WAIT_TIMEOUT_SECONDS - elapsed))
    if (( remaining <= 0 )); then
      progress_clear
      echo "Timed out waiting for $NAME; inspect training log: ${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}/$NAME.log" >&2
      return 1
    fi
    if [[ -z "${LIBERO_PROGRESS_FD:-}" && "$announced" == false ]]; then
      echo "Waiting for completed checkpoint: $NAME (timeout ${WAIT_TIMEOUT_SECONDS}s)" >&2
      announced=true
    fi
    progress_status "$LIBERO_PROGRESS_LABEL waiting for checkpoint $STEP_PADDED (${elapsed}s elapsed) ..."
    delay=$WAIT_INTERVAL_SECONDS
    if (( delay > remaining )); then delay=$remaining; fi
    sleep "$delay"
  done
}
run_one() {
  # A directory label must agree with the saved training factors.
  python - "$CHECKPOINT/pretrained_model/train_config.json" "$CASE" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
factors = config.get("pdet", {})
expected = dict(zip(("prompt_seed", "data_seed", "noise_seed", "time_seed"), map(int, sys.argv[2])))
if factors.get("enabled") is not True or any(factors.get(k) != v for k, v in expected.items()):
    raise SystemExit(f"PDET checkpoint factors do not match its run name: expected {expected}, got {factors}")
if config.get("seed") != 0:
    raise SystemExit("PDET checkpoint must retain global training seed 0")
PY
  VLM_PROMPT_TOKENS=2 RUN_NAME="$NAME" DRY_RUN="$1" \
    bash "$SCRIPT_DIR/run_eval_libero_vlm_only.sh" base trained "$SUITES" 0
}

INDEX=0
for CASE in "${CASES[@]}"; do
  INDEX=$((INDEX + 1))
  NAME="${RUN_PREFIX}-base-vlm2-P${CASE:0:1}-D${CASE:1:1}-E${CASE:2:1}-T${CASE:3:1}"
  CHECKPOINT="$CKPT_ROOT/$NAME/checkpoints/$STEP_PADDED"
  export LIBERO_PROGRESS_LABEL="[$INDEX/${#CASES[@]}] P${CASE:0:1}-D${CASE:1:1}-E${CASE:2:1}-T${CASE:3:1}"
  wait_for_checkpoint
  if [[ "$DRY_RUN" == true ]]; then run_one true; continue; fi
  mkdir -p "$EVAL_LOG_ROOT"
  LOG="$EVAL_LOG_ROOT/${NAME}-${STEP_PADDED}-${SUITE_TAG}-${STAMP}.log"
  progress_status "$LIBERO_PROGRESS_LABEL checking/loading model and environments ..."
  # Use a subshell with errexit; an `if run_one` would suppress errexit inside the function.
  set +e
  (set -e; run_one false) >"$LOG" 2>&1
  status=$?
  set -e
  progress_clear
  if (( status != 0 )); then
    echo "Evaluation/preflight failed: $NAME (exit=$status); log: $LOG" >&2
    exit "$status"
  fi
  RESULT="$BASE_OUTPUT/libero060-all-${NAME}-${STEP_PADDED}-${SUITE_TAG}-resume/eval_info.json"
  if SUMMARY=$(python - "$RESULT" "$NAME" "$STEP_PADDED" "$TASKS" "$EPISODES_PER_TASK" <<'PY' 2>>"$LOG"
import json
import sys
from pathlib import Path

path, name, step, suites_csv, episodes = sys.argv[1:]
expected = {(suite, task_id) for suite in suites_csv.split(",") for task_id in range(10)}
data = json.loads(Path(path).read_text())
tasks = data.get("per_task") or data.get("Aggregated Metrics for per_task")
if not isinstance(tasks, list):
    raise ValueError(f"Missing per-task results in {path}")
seen, by_suite = set(), {}
for task in tasks:
    key = (task["task_group"], task["task_id"])
    if key not in expected or key in seen:
        raise ValueError(f"Unexpected or duplicate task: {key}")
    seen.add(key)
    values = task.get("metrics", task).get("successes")
    if not isinstance(values, list) or len(values) != int(episodes) or any(type(v) is not bool for v in values):
        raise ValueError(f"Incomplete or invalid episode results: {key}")
    by_suite.setdefault(key[0], []).extend(values)
if seen != expected:
    raise ValueError(f"Missing tasks: {sorted(expected - seen)}")
print(f"===== {name} | checkpoint {step} =====")
total, success = 0, 0
for suite, values in sorted(by_suite.items()):
    n, wins = len(values), sum(values)
    total += n
    success += wins
    print(f"{suite}: {100 * wins / n:.2f}% ({wins}/{n})")
print(f"overall: {100 * success / total:.2f}% ({success}/{total})")
PY
  ); then
    printf '%s\n\n' "$SUMMARY"
    printf '\n%s\n' "$SUMMARY" >>"$LOG"
  else
    echo "Result validation failed: $NAME; log: $LOG" >&2
    exit 1
  fi
done
