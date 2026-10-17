#!/usr/bin/env bash
# Evaluate four separately trained policies on recorded data; no robot connection.
set -euo pipefail

TASK=all
DRY_RUN=false
for arg in "$@"; do
  case "${arg}" in
    all|1|2|3|4) TASK="${arg}" ;;
    --dry_run) DRY_RUN=true ;;
    -h|--help)
      cat <<'EOF'
Usage: bash examples/piper/eval_four_tasks.sh [all|1|2|3|4] [--dry_run]

Default: run tasks 1,2,3,4 sequentially; 256 observations and 6 PNGs per task.
Each PNG compares "teleoperation" with "action from model".
Run in the Python environment used for PI05 inference. No robot/RealSense access.

Settings (environment variables):
  PLOTS_PER_TASK=6         Number of evenly selected plots per task
  MAX_SAMPLES=256          Maximum evaluated observations; 0 = all candidates
  EPISODES=all             Training episode IDs, e.g. 0,1,2,3,4
  STRIDE=8                 Candidate observation interval within each episode
  SEED=0                   Flow noise seed
  DEVICE=cuda              PyTorch device
  PIPER_DATASET_BASE        Parent of the four task datasets
  PIPER_MODEL_BASE          Parent of task 1/2/3 model run directories
  PIPER_POLICY_1 .. _4      Override any one full pretrained_model path
  PIPER_TOKENIZER_PATH      Local tokenizer directory
  PIPER_OUTPUT_ROOT         Default: <repo>/outputs/piper-offline/four-tasks-<timestamp>

--dry_run prints commands only; it does not validate files or run inference.
Normal execution checks all selected input paths before loading the first model.
Existing task result directories are never overwritten. Select a new output root
for a repeat run, or select just the unfinished task in an existing output root.
EOF
      exit 0 ;;
    *) echo "Unknown argument: ${arg}; use --help" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PIPER_DISK=/media/drx/a1ea95d8-943f-4c65-b4a3-05f5927573b7
PIPER_DATASET_BASE="${PIPER_DATASET_BASE:-${PIPER_DISK}/dataset/May-pick-and-place}"
PIPER_MODEL_BASE="${PIPER_MODEL_BASE:-${PIPER_DISK}/Wu_Yinan/Priming}"
PIPER_TOKENIZER_PATH="${PIPER_TOKENIZER_PATH:-/home/drx/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}"
PIPER_OUTPUT_ROOT="${PIPER_OUTPUT_ROOT:-${REPO_ROOT}/outputs/piper-offline/four-tasks-$(date +%Y%m%d-%H%M%S)}"

TASK_NAMES=(
  1-put_the_apple_on_the_yellow_plate
  2-remove_the_cuboid_from_blue_plate
  3-move_the_tennis_from_yellow_plate_to_blue_plate
  4-pick_the_red_cube_into_the_yellow_plate
)
POLICIES=(
  "${PIPER_POLICY_1:-${PIPER_MODEL_BASE}/pi05-may-${TASK_NAMES[0]}-full_reference-seed0/012000/pretrained_model}"
  "${PIPER_POLICY_2:-${PIPER_MODEL_BASE}/pi05-may-${TASK_NAMES[1]}-full_reference-seed0/012000/pretrained_model}"
  "${PIPER_POLICY_3:-${PIPER_MODEL_BASE}/pi05-may-${TASK_NAMES[2]}-full_reference-seed0/pretrained_model}"
  "${PIPER_POLICY_4:-/home/drx/Downloads/wyn/pi05-may-${TASK_NAMES[3]}-full_reference-seed0/012000/pretrained_model}"
)
SELECTED=(0 1 2 3)
if [[ "${TASK}" != all ]]; then
  SELECTED=("$((TASK - 1))")
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [[ "${DRY_RUN}" == false ]]; then
  # Check all tasks first, so a typo in task 4 does not waste tasks 1..3 GPU time.
  [[ -f "${PIPER_TOKENIZER_PATH}/tokenizer_config.json" ]] || {
    echo "Missing tokenizer_config.json: ${PIPER_TOKENIZER_PATH}" >&2; exit 2;
  }
  for i in "${SELECTED[@]}"; do
    for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
      [[ -f "${POLICIES[i]}/${file}" ]] || {
        echo "Task $((i + 1)): missing checkpoint file ${POLICIES[i]}/${file}" >&2; exit 2;
      }
    done
    for entry in meta/info.json data videos; do
      [[ -e "${PIPER_DATASET_BASE}/${TASK_NAMES[i]}/${entry}" ]] || {
        echo "Task $((i + 1)): missing dataset entry ${PIPER_DATASET_BASE}/${TASK_NAMES[i]}/${entry}" >&2; exit 2;
      }
    done
    [[ ! -e "${PIPER_OUTPUT_ROOT}/${TASK_NAMES[i]}" ]] || {
      echo "Results already exist: ${PIPER_OUTPUT_ROOT}/${TASK_NAMES[i]}; set a new PIPER_OUTPUT_ROOT" >&2
      exit 2
    }
  done
  mkdir -p -- "${PIPER_OUTPUT_ROOT}"
else
  echo "COMMANDS_ONLY: paths are not checked; no inference or output files."
fi

echo "Output root: ${PIPER_OUTPUT_ROOT}"
for i in "${SELECTED[@]}"; do
  COMMAND=(
    python "${REPO_ROOT}/eval_piper_offline.py"
    --policy_path "${POLICIES[i]}"
    --dataset_root "${PIPER_DATASET_BASE}/${TASK_NAMES[i]}"
    --tokenizer_path "${PIPER_TOKENIZER_PATH}"
    --episodes "${EPISODES:-all}" --stride "${STRIDE:-8}"
    --max_samples "${MAX_SAMPLES:-256}" --execution_steps 8
    --seed "${SEED:-0}" --device "${DEVICE:-cuda}"
    --plots "${PLOTS_PER_TASK:-6}"
    --output_dir "${PIPER_OUTPUT_ROOT}/${TASK_NAMES[i]}"
  )
  printf 'Task %s: ' "$((i + 1))"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == false ]]; then
    PYTHONUNBUFFERED=1 "${COMMAND[@]}" 2>&1 | tee -a "${PIPER_OUTPUT_ROOT}/${TASK_NAMES[i]}.log"
  fi
done
if [[ "${DRY_RUN}" == false ]]; then
  echo "BATCH_EVAL_OK: ${PIPER_OUTPUT_ROOT}"
fi
