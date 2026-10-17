#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_vlm_only_token_sweep.sh [base|finetuned|both] [all|no10|SUITE] [SEED]
Defaults: both all 0, VLM_PROMPT_TOKEN_COUNTS="2 4", CHECKPOINT_STEP=3000.
Order: base-vlm2, base-vlm4, finetuned-vlm2, finetuned-vlm4; trained checkpoints only.
All checkpoints pass the existing VLM-only preflight before any GPU evaluation starts.
Runs sequentially on GPU_ID=0 (override to 1); stop on the first failure.
Same evaluation protocol: EVAL_SEED=1000, 10 episodes/task, n_action_steps=10.

Terminal output shows one in-place episode progress bar, then final success rates.
The bar shows model index, completed/total episodes, elapsed time and this model's ETA.
PROGRESS=auto (default) shows it in a terminal; true forces it, false hides it.
Full output (including errors/video paths) is saved under LOG_ROOT/eval
(LOG_ROOT defaults to /data1/wyn/chkpt/2601-lerobot/logs);
EVAL_LOG_ROOT can override that log directory. On failure, print the log path.
Existing per-task resume checks still apply. Videos are still saved as before.
DRY_RUN=true checks all models and prints commands without starting evaluation or writing logs.

CKPT_ROOT defaults to /data1/wyn/chkpt/2601-lerobot/PromptVLA
BASE_OUTPUT defaults to CKPT_ROOT/eval
Input models/tokenizer default to /home/wyn/models.
Other overrides: BASE_PRETRAINED_PATH, FINETUNED_PRETRAINED_PATH, TOKENIZER_PATH,
RUN_PREFIX (pi05), VLM_PROMPT_TOKEN_COUNTS, EPISODES_PER_TASK, EVAL_SEED, CHECKPOINT_STEP.
Unset RUN_NAME, PRETRAINED_PATH, VLM_PROMPT_TOKENS, BASE_CKPT and BASE_MODEL_PATH.

GPU_ID=1 bash run_eval_libero_vlm_only_token_sweep.sh both all 0
VLM_PROMPT_TOKEN_COUNTS=4 GPU_ID=1 bash run_eval_libero_vlm_only_token_sweep.sh finetuned all 0
EOF
  exit 0
fi
if (( $# > 3 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
SELECT="${1:-both}"; SUITES="${2:-all}"; SEED="${3:-0}"
case "$SELECT" in
  both) SOURCES=(base finetuned) ;;
  base|finetuned) SOURCES=("$SELECT") ;;
  *) echo "Invalid source; see --help" >&2; exit 2 ;;
esac
case "$SUITES" in
  all) SUITE_TAG=libero-all4; TASKS=libero_spatial,libero_object,libero_goal,libero_10 ;;
  no10) SUITE_TAG=libero-no10; TASKS=libero_spatial,libero_object,libero_goal ;;
  libero_spatial|libero_object|libero_goal|libero_10) SUITE_TAG="$SUITES"; TASKS="$SUITES" ;;
  *) echo "Invalid suite; see --help" >&2; exit 2 ;;
esac
for key in RUN_NAME PRETRAINED_PATH VLM_PROMPT_TOKENS BASE_CKPT BASE_MODEL_PATH; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key for this sweep; see --help" >&2; exit 2; fi
done
read -r -a TOKEN_COUNTS <<< "${VLM_PROMPT_TOKEN_COUNTS-2 4}"
if (( ${#TOKEN_COUNTS[@]} == 0 )); then echo "VLM_PROMPT_TOKEN_COUNTS cannot be empty" >&2; exit 2; fi
declare -A SEEN_COUNTS=()
for count in "${TOKEN_COUNTS[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid VLM token count: $count" >&2; exit 2; fi
  if [[ -n "${SEEN_COUNTS[$count]:-}" ]]; then echo "Duplicate VLM token count: $count" >&2; exit 2; fi
  SEEN_COUNTS["$count"]=1
done
DRY_RUN="${DRY_RUN:-false}"
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then echo "Invalid DRY_RUN" >&2; exit 2; fi
PROGRESS="${PROGRESS:-auto}"
case "$PROGRESS" in auto|true|false) ;; *) echo "Invalid PROGRESS: use auto, true or false" >&2; exit 2 ;; esac
RUN_PREFIX="${RUN_PREFIX:-pi05}"
export GPU_ID="${GPU_ID:-0}" CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}" EVAL_SEED="${EVAL_SEED:-1000}"
for value in "$CHECKPOINT_STEP" "$EPISODES_PER_TASK"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Step/episodes must be positive integers" >&2; exit 2; fi
done
for value in "$SEED" "$EVAL_SEED"; do
  if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid seed" >&2; exit 2; fi
done
export BASE_PRETRAINED_PATH="${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}"
export FINETUNED_PRETRAINED_PATH="${FINETUNED_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_finetuned_v044}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export BASE_OUTPUT="${BASE_OUTPUT:-$CKPT_ROOT/eval}"
export TQDM_DISABLE=1
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}/eval}"
STEP_PADDED=$(printf '%06d' "$CHECKPOINT_STEP")
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
TOTAL_MODELS=$((${#SOURCES[@]} * ${#TOKEN_COUNTS[@]}))
# Keep a dedicated terminal stream across nested shell/Python log redirections.
# Only the overall bar writes here; warnings, inner bars and video paths stay in logs.
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

run_one() {
  local source="$1" count="$2" dry="$3"
  VLM_PROMPT_TOKENS="$count" RUN_NAME="${RUN_PREFIX}-${source}-vlm${count}-seed${SEED}" DRY_RUN="$dry" \
    bash "$SCRIPT_DIR/run_eval_libero_vlm_only.sh" "$source" trained "$SUITES" "$SEED"
}
select_run() {
  NAME="${RUN_PREFIX}-$1-vlm$2-seed${SEED}"
  LOG="$EVAL_LOG_ROOT/${NAME}-${STEP_PADDED}-${SUITE_TAG}-${STAMP}.log"
}

# Check the entire sweep before starting the first expensive rollout.
if [[ "$DRY_RUN" != true ]]; then mkdir -p "$EVAL_LOG_ROOT"; fi
MODEL_INDEX=0
for source in "${SOURCES[@]}"; do
  for count in "${TOKEN_COUNTS[@]}"; do
    MODEL_INDEX=$((MODEL_INDEX + 1))
    select_run "$source" "$count"
    progress_status "Checking [$MODEL_INDEX/$TOTAL_MODELS] $source-vlm$count ..."
    if [[ "$DRY_RUN" == true ]]; then
      run_one "$source" "$count" true
    elif run_one "$source" "$count" true >"$LOG" 2>&1; then
      :
    else
      status=$?
      progress_clear
      echo "Preflight failed: $NAME (exit=$status); log: $LOG" >&2
      exit "$status"
    fi
  done
done
if [[ "$DRY_RUN" == true ]]; then exit 0; fi

MODEL_INDEX=0
for source in "${SOURCES[@]}"; do
  for count in "${TOKEN_COUNTS[@]}"; do
    MODEL_INDEX=$((MODEL_INDEX + 1))
    select_run "$source" "$count"
    export LIBERO_PROGRESS_LABEL="[$MODEL_INDEX/$TOTAL_MODELS] $source-vlm$count"
    progress_status "$LIBERO_PROGRESS_LABEL loading model/environments ..."
    if run_one "$source" "$count" false >>"$LOG" 2>&1; then
      progress_clear
    else
      status=$?
      progress_clear
      echo "Evaluation failed: $NAME (exit=$status); log: $LOG" >&2
      exit "$status"
    fi
    RESULT="$BASE_OUTPUT/libero060-all-${NAME}-${STEP_PADDED}-${SUITE_TAG}-resume/eval_info.json"
    # Read structured results only after successful evaluation. Never scrape appended
    # summary logs, which can contain old results and repeated video-path dictionaries.
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
done
