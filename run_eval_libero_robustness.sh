#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash run_eval_libero_robustness.sh pro|plus [all|libero_spatial|libero_object|libero_goal|libero_10]
Default model: H2E/e30-h70/S0/steps6000/aexpert/r1, checkpoint006000; GPU_ID=0.
Requires a local complete official fork: BENCHMARK_ROOT (default /home/wyn/VLAA/LIBERO-PRO or LIBERO-plus).
Each fork needs its own bddl_files, init_files and package-relative assets; see README_libero_robustness.md.
Does NOT uninstall packages or modify ~/.libero. It selects the fork only in the evaluation process.
PLUS: all 10030 variants, exactly 1 episode each. PRO: 5 dimensions x 40 tasks, default10 episodes/task.
PRO_DIMENSIONS=all or comma-separated object,swap,lan,task,env; no missing dimensions silently skipped.
Controls: RUN_NAME, CKPT_ROOT, CHECKPOINT_STEP=6000, TOKENIZER_PATH, GPU_ID, EVAL_SEED=1000,
          BASE_OUTPUT, EVAL_LOG_ROOT, EPISODES_PER_TASK (PRO only), PROGRESS=true, DRY_RUN=false.
TRAIN_SCOPE=joint|expert_only|prompt_only (default joint); validate the saved training scope.
Independent suite invocations can run on different GPUs. Full checkpoints + saved processors are used.
Outputs: one progress bar, suite/category success rates, task-level resume; no videos.
EOF
  exit 0
fi
(( $# >= 1 && $# <= 2 )) || { echo 'Choose pro or plus; see --help' >&2; exit 2; }
KIND="$1"; MODE="${2:-all}"
case "$KIND" in pro|plus) ;; *) echo 'Invalid benchmark' >&2; exit 2 ;; esac
case "$MODE" in all|libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid base suite' >&2; exit 2 ;; esac
export RUN_NAME="${RUN_NAME:-pi05-curriculum-H2E-e30-h70-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-6000}" EVAL_SEED="${EVAL_SEED:-1000}"
export GPU_ID="${GPU_ID:-0}" TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
PROGRESS="${PROGRESS:-true}"; DRY_RUN="${DRY_RUN:-false}"
TRAIN_SCOPE="${TRAIN_SCOPE:-joint}"
case "$TRAIN_SCOPE" in joint|expert_only|prompt_only) ;; *) echo 'Invalid TRAIN_SCOPE' >&2; exit 2 ;; esac
for value in "$PROGRESS" "$DRY_RUN"; do
  case "$value" in true|false) ;; *) echo 'PROGRESS/DRY_RUN must be true/false' >&2; exit 2 ;; esac
done
[[ "$CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ && "$EVAL_SEED" =~ ^(0|[1-9][0-9]*)$ && "$GPU_ID" =~ ^[0-9]+$ ]] || {
  echo 'Invalid checkpoint step, seed or single GPU_ID' >&2; exit 2;
}
if [[ "$KIND" == plus ]]; then
  BENCHMARK_ROOT="${BENCHMARK_ROOT:-/home/wyn/VLAA/LIBERO-plus}"
  EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
  [[ "$EPISODES_PER_TASK" == 1 ]] || { echo 'PLUS requires EPISODES_PER_TASK=1' >&2; exit 2; }
else
  BENCHMARK_ROOT="${BENCHMARK_ROOT:-/home/wyn/VLAA/LIBERO-PRO}"
  EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
fi
[[ "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EPISODES_PER_TASK' >&2; exit 2; }
PRO_DIMENSIONS="${PRO_DIMENSIONS:-all}"
[[ "$PRO_DIMENSIONS" =~ ^[a-z,]+$ ]] || { echo 'Invalid PRO_DIMENSIONS' >&2; exit 2; }
export EPISODES_PER_TASK DRY_RUN
export BASE_OUTPUT="${BASE_OUTPUT:-$CKPT_ROOT/eval/robustness}/$KIND/seed${EVAL_SEED}-ep${EPISODES_PER_TASK}-dims${PRO_DIMENSIONS//,/-}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs/eval}/robustness/$KIND"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1
unset MUJOCO_EGL_DEVICE_ID LIBERO_PROGRESS_FD LIBERO_PROGRESS_LABEL
unset LEROBOT_EVAL_BASE LEROBOT_EVAL_ATTENTION BASE_MODEL_PATH VLM_PROMPT_TOKENS
export LEROBOT_EVAL_FINETUNED_PROMPT=1 LIBERO_BENCHMARK="$KIND"
STEP_PADDED=$(printf '%06d' "$CHECKPOINT_STEP")
export BASE_CKPT="$CKPT_ROOT/$RUN_NAME/checkpoints"
CHECKPOINT="$BASE_CKPT/$STEP_PADDED/pretrained_model"
PROTOCOL_DIR="$BASE_OUTPUT/protocol-$MODE"
export LIBERO_ROBUSTNESS_MANIFEST="$PROTOCOL_DIR/manifest.json"
export LIBERO_CONFIG_PATH="$PROTOCOL_DIR/config"
[[ -d "$TOKENIZER_PATH" ]] || { echo "Missing tokenizer: $TOKENIZER_PATH" >&2; exit 1; }
mkdir -p "$EVAL_LOG_ROOT"
LOG="$EVAL_LOG_ROOT/${RUN_NAME}-${STEP_PADDED}-${MODE}-seed${EVAL_SEED}-ep${EPISODES_PER_TASK}-dims${PRO_DIMENSIONS//,/-}.log"
echo "Checking $KIND resources and checkpoint $STEP_PADDED; log=$LOG"
if ! {
  python -m lerobot.scripts.libero_robustness_checkpoint "$CHECKPOINT" --scope "$TRAIN_SCOPE" &&
  python -m lerobot.scripts.libero_robustness prepare --root "$BENCHMARK_ROOT" --directory "$PROTOCOL_DIR" \
    --benchmark "$KIND" --mode "$MODE" --dimensions "$PRO_DIMENSIONS" --episodes "$EPISODES_PER_TASK" --seed "$EVAL_SEED"
} >"$LOG" 2>&1; then
  tail -n 50 "$LOG" >&2; exit 1
fi
if [[ "$DRY_RUN" == true ]]; then
  bash "$SCRIPT_DIR/run_eval_libero-full.sh" vlm_only "$MODE" 0
  exit 0
fi
if [[ "$PROGRESS" == true ]]; then
  exec 3>&2
  export LIBERO_PROGRESS_FD=3 LIBERO_PROGRESS_LABEL="$KIND/$MODE"
fi
if ! bash "$SCRIPT_DIR/run_eval_libero-full.sh" vlm_only "$MODE" 0 >>"$LOG" 2>&1; then
  tail -n 70 "$LOG" >&2; exit 1
fi
TAG="$MODE"; [[ "$MODE" != all ]] || TAG=libero-all4
RESULT="$BASE_OUTPUT/libero060-all-${RUN_NAME}-${STEP_PADDED}-${KIND}-${TAG}-resume/eval_info.json"
echo "===== $RUN_NAME | $KIND | checkpoint $STEP_PADDED ====="
python -m lerobot.scripts.libero_robustness summarize "$LIBERO_ROBUSTNESS_MANIFEST" "$RESULT"
