#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh [plots|eval|align|report|all] [SEED=1]
Default: all 1; base VLM-only tokens 1 2 4 8 16 32, checkpoint 3000.
  plots  CPU-only histories and 0/1500/3000 spectra from existing prompt_rank files.
  eval   Six sequential four-suite LIBERO evaluations; one terminal progress bar.
  align  Replay 0/1500/3000 prompts on one GPU; no retraining or new full checkpoints.
  report Require completed eval and alignment, draw all figures and join success rates.
  all    eval -> align -> report. Stops on failure; completed eval/tasks and align/steps resume.

Defaults: GPU_ID=0, SOURCE=base, VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32",
CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA,
BASE_OUTPUT=CKPT_ROOT/eval, REPORT_ROOT=CKPT_ROOT/analysis,
ALIGNMENT_STEPS="0 1500 3000", CHECKPOINT_STEP=3000, EVAL_SEED=1000, EPISODES_PER_TASK=10.
ALIGNMENT_NAME=prompt_alignment, HELDOUT_PROBES=4, HELDOUT_SEED=2027, RIDGE_RELATIVE=0.01.
Set FONT_PATH for a specific figure font. RUN_PREFIX=pi05. SOURCE may also be finetuned.
DRY_RUN=true prints dispatch commands without loading data, running jobs or creating outputs.

GPU_ID=1 bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh all 1
bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh plots 1
EOF
  exit 0
fi
if (( $# > 2 )); then echo "Too many arguments; see --help" >&2; exit 2; fi
MODE="${1:-all}"; SEED="${2:-1}"
case "$MODE" in plots|eval|align|report|all) ;; *) echo "Invalid mode" >&2; exit 2 ;; esac
export SOURCE="${SOURCE:-base}"
case "$SOURCE" in base|finetuned) ;; *) echo "Invalid SOURCE" >&2; exit 2 ;; esac
export VLM_PROMPT_TOKEN_COUNTS="${VLM_PROMPT_TOKEN_COUNTS-1 2 4 8 16 32}"
read -r -a COUNTS <<< "$VLM_PROMPT_TOKEN_COUNTS"
read -r -a STEPS <<< "${ALIGNMENT_STEPS-0 1500 3000}"
if (( ${#COUNTS[@]} == 0 || ${#STEPS[@]} == 0 )); then echo "Counts/steps cannot be empty" >&2; exit 2; fi
for value in "$SEED" "${STEPS[@]}"; do
  if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "Invalid seed/step" >&2; exit 2; fi
done
declare -A SEEN=()
for value in "${COUNTS[@]}"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ || -n "${SEEN[$value]:-}" ]]; then
    echo "Invalid/duplicate token count" >&2; exit 2
  fi
  SEEN["$value"]=1
done
export CKPT_ROOT="${CKPT_ROOT:-/data1/wyn/chkpt/2601-lerobot/PromptVLA}"
export BASE_OUTPUT="${BASE_OUTPUT:-$CKPT_ROOT/eval}"
export GPU_ID="${GPU_ID:-0}" RUN_PREFIX="${RUN_PREFIX:-pi05}" CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
export EVAL_SEED="${EVAL_SEED:-1000}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
REPORT_ROOT="${REPORT_ROOT:-$CKPT_ROOT/analysis}"
OUTPUT="$REPORT_ROOT/${RUN_PREFIX}-${SOURCE}-seed${SEED}"
ALIGNMENT_NAME="${ALIGNMENT_NAME:-prompt_alignment}"
DRY_RUN="${DRY_RUN:-false}"
case "$DRY_RUN" in true|false) ;; *) echo "Invalid DRY_RUN" >&2; exit 2 ;; esac
dispatch() {
  if [[ "$DRY_RUN" == true ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
COMMON=(--ckpt-root "$CKPT_ROOT" --tokens "${COUNTS[@]}" --seed "$SEED" --source "$SOURCE"
        --run-prefix "$RUN_PREFIX" --steps "${STEPS[@]}" --analysis-name "$ALIGNMENT_NAME")
PLOT=(python -m lerobot.scripts.plot_pi05_prompt_rank "${COMMON[@]}" --final-step "$CHECKPOINT_STEP")
if [[ -n "${FONT_PATH:-}" ]]; then PLOT+=(--font-path "$FONT_PATH"); fi
if [[ "$MODE" == plots ]]; then
  dispatch "${PLOT[@]}" --output-dir "$OUTPUT/spectra"
fi
if [[ "$MODE" == eval || "$MODE" == all ]]; then
  dispatch bash "$REPO_ROOT/run_eval_libero_vlm_only_token_sweep.sh" "$SOURCE" all "$SEED"
fi
if [[ "$MODE" == align || "$MODE" == all ]]; then
  dispatch env CUDA_VISIBLE_DEVICES="$GPU_ID" python -m lerobot.scripts.analyze_pi05_prompt_alignment \
    "${COMMON[@]}" --checkpoint-step "$CHECKPOINT_STEP" --device cuda \
    --heldout-probes "${HELDOUT_PROBES:-4}" --heldout-seed "${HELDOUT_SEED:-2027}" \
    --ridge-relative "${RIDGE_RELATIVE:-0.01}"
fi
if [[ "$MODE" == report || "$MODE" == all ]]; then
  dispatch "${PLOT[@]}" --output-dir "$OUTPUT/complete" --eval-root "$BASE_OUTPUT" \
    --eval-seed "$EVAL_SEED" --episodes-per-task "$EPISODES_PER_TASK" --include-alignment
fi
