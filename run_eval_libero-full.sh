#!/bin/bash
set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Usage: bash run_eval_libero-full.sh VARIANT [all|no10|SUITE] [TRAINING_SEED]
# all: four standard LIBERO suites; no10: spatial/object/goal only.
usage() {
  echo "Usage: $0 VARIANT [all|no10|libero_spatial|libero_object|libero_goal|libero_10] [SEED]"
  echo "Example: $0 direct_dual no10 0"
  echo "VARIANT: full_reference, no_bridge, direct_dual, dual_prompt_only, vlm_only, action_only, no_cabo, pi05_libero_base"
  echo "Defaults: all suites, training seed 0, checkpoint 3000, 10 episodes/task."
  echo "VLM_PROMPT_TOKENS=8 bash $0 vlm_only all 0  # token-sweep checkpoint"
  echo "With VLM_PROMPT_TOKENS, legacy run names are pi05-vlm_only-vlmN-seedS; set CKPT_ROOT for legacy outputs."
  echo "For native VLM-only runs use run_eval_libero_vlm_only.sh or run_eval_libero_vlm_only_token_sweep.sh."
  echo "Overrides: CKPT_ROOT, BASE_CKPT, BASE_OUTPUT, RUN_NAME, CHECKPOINT_STEP (0 allowed), EPISODES_PER_TASK, GPU_ID, EVAL_SEED."
  echo "Results default to CKPT_ROOT/eval; logs to EVAL_LOG_ROOT (default LOG_ROOT/eval)."
  echo "DRY_RUN=true validates paths/token counts and prints commands without loading a model."
  echo "Evaluate all 1/2/8/16-token models: bash run_eval_libero_vlm_token_sweep.sh all 0"
  echo "Evaluate the base with BOTH prompt banks disabled: bash run_eval_libero_base.sh all"
  echo "For pi05_libero_base use BASE_MODEL_PATH (a model directory directly, not checkpoints/003000)."
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi
VARIANT="$1"
MODE="${2:-all}"
SEED="${3:-0}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
VLM_PROMPT_TOKENS="${VLM_PROMPT_TOKENS:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
DRY_RUN="${DRY_RUN:-false}"
case "$VARIANT" in
  full_reference|no_bridge|direct_dual|dual_prompt_only|vlm_only|action_only|no_cabo|pi05_libero_base) ;;
  *) echo "Unknown model variant: $VARIANT" >&2; usage >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer without leading zeros" >&2
  exit 2
fi
if [[ ! "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES_PER_TASK must be a positive integer" >&2
  exit 2
fi
if [[ ! "$CHECKPOINT_STEP" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "CHECKPOINT_STEP must be a non-negative integer without leading zeros" >&2
  exit 2
fi
EVAL_SEED="${EVAL_SEED:-}"
if [[ -n "$EVAL_SEED" && ! "$EVAL_SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "EVAL_SEED must be a non-negative integer without leading zeros" >&2
  exit 2
fi
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then
  echo "DRY_RUN must be true or false" >&2
  exit 2
fi
DEFAULT_RUN_NAME="pi05-${VARIANT}-seed${SEED}"
TRAINING_SEED_LABEL="$SEED"
DEFAULT_CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA
export LEROBOT_EVAL_BASE=0
if [[ "$VARIANT" == pi05_libero_base ]]; then
  if [[ -n "${BASE_CKPT:-}" || -n "$VLM_PROMPT_TOKENS" ]]; then
    echo "Base evaluation uses BASE_MODEL_PATH and zero prompts; unset BASE_CKPT and VLM_PROMPT_TOKENS" >&2
    exit 2
  fi
  DEFAULT_RUN_NAME=pi05_libero_base-no_prompt
  TRAINING_SEED_LABEL=none
  BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PRETRAINED_PATH:-${BASE_PRETRAINED_PATH:-/home/wyn/models/lerobot/pi05_libero_base}}}"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wyn/models/paligemma-3b-pt-224}"
  export LEROBOT_EVAL_BASE=1
  export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "$VLM_PROMPT_TOKENS" ]]; then
  if [[ "$VARIANT" != vlm_only || ! "$VLM_PROMPT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "VLM_PROMPT_TOKENS requires vlm_only and a positive integer without leading zeros" >&2
    exit 2
  fi
  DEFAULT_RUN_NAME="pi05-vlm_only-vlm${VLM_PROMPT_TOKENS}-seed${SEED}"
fi
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
case "$MODE" in
  all)
    TASKS="libero_spatial,libero_object,libero_goal,libero_10"
    SUITE_TAG="libero-all4"
    ;;
  no10|--skip-libero-10)
    TASKS="libero_spatial,libero_object,libero_goal"
    SUITE_TAG="libero-no10"
    ;;
  libero_spatial|libero_object|libero_goal|libero_10)
    TASKS="$MODE"
    SUITE_TAG="$MODE"
    ;;
  -h|--help)
    usage
    echo "all (default): spatial, object, goal, libero_10"
    echo "no10: spatial, object, goal"
    exit 0
    ;;
  *) echo "Unknown mode: $MODE; use all, no10, or a LIBERO suite name" >&2; exit 2 ;;
esac
# Activate your existing lerobot environment before running.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset MUJOCO_EGL_DEVICE_ID || true
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/wyn/.libero}"
export PYTHONUNBUFFERED=1

CKPT_ROOT="${CKPT_ROOT:-${DEFAULT_CKPT_ROOT}}"
# Explicit BASE_CKPT overrides automatic checkpoint discovery only.
BASE_CKPT="${BASE_CKPT:-${CKPT_ROOT}/${RUN_NAME}/checkpoints}"
BASE_OUTPUT="${BASE_OUTPUT:-$CKPT_ROOT/eval}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${LOG_ROOT:-/data1/wyn/chkpt/2601-lerobot/logs}/eval}"
STEPS=("$CHECKPOINT_STEP")
if [[ "$VARIANT" == pi05_libero_base ]]; then
  STEPS=(base)
fi
SUMMARY_LOG="$EVAL_LOG_ROOT/eval_${RUN_NAME}-${SUITE_TAG}-summary.log"
if [[ "$DRY_RUN" != true ]]; then
  mkdir -p "$BASE_OUTPUT" "$EVAL_LOG_ROOT"
  touch "$SUMMARY_LOG"
fi
FINAL_EXIT=0

for STEP in "${STEPS[@]}"; do
  if [[ "$VARIANT" == pi05_libero_base ]]; then
    STEP_PADDED=base
    CKPT="$BASE_MODEL_PATH"
  else
    STEP_PADDED=$(printf "%06d" "$STEP")
    CKPT="$BASE_CKPT/$STEP_PADDED/pretrained_model"
  fi
  OUTPUT_DIR="$BASE_OUTPUT/libero060-all-${RUN_NAME}-${STEP_PADDED}-${SUITE_TAG}-resume"
  LOG="$EVAL_LOG_ROOT/eval_${RUN_NAME}-${STEP_PADDED}-${SUITE_TAG}.log"

  if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    FINAL_EXIT=1
    continue
  fi
  if [[ "$VARIANT" == pi05_libero_base ]]; then
    if ! python "$SCRIPT_DIR/src/lerobot/scripts/libero_base_eval.py" "$CKPT" "${TOKENIZER_PATH:-}"; then
      FINAL_EXIT=1
      continue
    fi
  fi
  if [[ -n "$VLM_PROMPT_TOKENS" ]]; then
    # Read saved architecture; never resize/override learned prompts at evaluation.
    if ! python - "$CKPT/config.json" "$VLM_PROMPT_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
expected = int(sys.argv[2])
actual = config.get("num_vlm_prompt_tokens")
action = config.get("num_prompt_tokens")
if type(actual) is not int or actual != expected or type(action) is not int or action != 0:
    raise SystemExit(
        f"Checkpoint prompt mismatch: {path}: expected VLM={expected}, action=0; "
        f"found VLM={actual}, action={action}"
    )
PY
    then
      FINAL_EXIT=1
      continue
    fi
  fi
  EVAL_ARGS=(
    --policy.path="$CKPT"
    --output_dir="$OUTPUT_DIR"
    --env.type=libero
    --env.task="$TASKS"
    --env.control_mode=relative
    --env.max_parallel_tasks=1
    --eval.batch_size=1
    --eval.n_episodes="$EPISODES_PER_TASK"
    --policy.n_action_steps=10
    --policy.use_amp=false
    --policy.device=cuda
    --policy.compile_model=false
    --policy.gradient_checkpointing=false
  )
  if [[ -n "$EVAL_SEED" ]]; then
    EVAL_ARGS+=(--seed="$EVAL_SEED")
  fi
  if [[ "$VARIANT" == pi05_libero_base ]]; then
    EVAL_ARGS+=(
      --policy.num_vlm_prompt_tokens=0
      --policy.num_prompt_tokens=0
      --policy.cabo_enabled=false
      --policy.training_stage=flow
      --policy.next_action_pretrain_steps=0
      --policy.next_action_bridge_steps=0
      --policy.train_action_projections=false
      --policy.use_peft=false
    )
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
      EVAL_ARGS+=(--policy.tokenizer_name="$TOKENIZER_PATH")
    fi
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "model=$RUN_NAME training_seed=$TRAINING_SEED_LABEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "output_dir=$OUTPUT_DIR"
    echo "log=$LOG"
    printf '%q ' python -u - "${EVAL_ARGS[@]}"
    printf '\n'
    continue
  fi
  touch "$LOG"
  {
    echo "===== evaluating checkpoint $STEP_PADDED at $(date) ====="
    echo "model=$RUN_NAME training_seed=$TRAINING_SEED_LABEL"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "mode=$MODE suites=$TASKS episodes_per_task=$EPISODES_PER_TASK"
    echo "policy=$CKPT"
    echo "output_dir=$OUTPUT_DIR"
    echo "log=$LOG"
  } | tee -a "$LOG"

  set +e
  # Pass arguments normally; stdin contains only the progress wrapper.
  python -u - "${EVAL_ARGS[@]}" \
    <<'PY' 2>&1 | tee -a "$LOG"
import importlib
import importlib.metadata
import inspect
import sys
import os
import json
import hashlib
import tempfile
import fcntl
from pathlib import Path

# Fail early: test actual CUDA allocation and computation in this same process.
import torch
print("Python:", sys.executable, flush=True)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; stopping evaluation instead of falling back to CPU. Check GPU_ID and PyTorch.")
try:
    probe = torch.ones((32, 32), device="cuda")
    assert (probe @ probe).sum().item() == 32768
    torch.cuda.synchronize()
    print("CUDA check passed:", torch.cuda.get_device_name(0), flush=True)
    del probe
except Exception as error:
    raise RuntimeError("CUDA computation preflight failed") from error

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# The sweep keeps ordinary stdout/stderr in its log and exposes only this stream
# to the terminal. Existing direct invocations keep their original progress style.
progress_fd = os.environ.get("LIBERO_PROGRESS_FD")
progress_stream = os.fdopen(os.dup(int(progress_fd)), "w", buffering=1) if progress_fd else sys.stderr

entries = list(importlib.metadata.entry_points(group="console_scripts", name="lerobot-eval"))
if len(entries) != 1:
    raise RuntimeError("Cannot uniquely resolve lerobot-eval in this Python environment")
entry = entries[0]
module = importlib.import_module(entry.module)
base_manifest = {}
if os.environ.get("LEROBOT_EVAL_BASE") == "1":
    from lerobot.scripts.libero_base_eval import install_base_evaluation
    base_manifest = install_base_evaluation(module, os.environ.get("TOKENIZER_PATH"))
elif os.environ.get("LEROBOT_EVAL_FINETUNED_PROMPT") == "1":
    from lerobot.scripts.libero_finetuned_prompt import install_finetuned_evaluation
    base_manifest = install_finetuned_evaluation(module, os.environ.get("TOKENIZER_PATH"))
attention_manifest = {}
if os.environ.get("LEROBOT_EVAL_ATTENTION") == "1":
    from lerobot.scripts.libero_attention import install_attention_evaluation, require_saved_attention_videos
    attention_manifest = install_attention_evaluation(module)
original_all = module.eval_policy_all
original_one = module.run_one
original_rollout = module.rollout
all_signature = inspect.signature(original_all)
one_signature = inspect.signature(original_one)
def argument(name):
    return next(x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith(name + "="))

output = Path(argument("--output_dir")).resolve()
model = Path(argument("--policy.path")).resolve()
output.mkdir(parents=True, exist_ok=True)
lock = (output / ".resume.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
records = output / "task_results"
records.mkdir(exist_ok=True)

def atomic_json(path, data):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

manifest = {
    "version": 1, "arguments": sys.argv[1:],
    "model_files": [[str(p.relative_to(model)), p.stat().st_size, p.stat().st_mtime_ns]
                    for p in sorted(model.rglob("*")) if p.is_file()],
    "evaluator_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    **attention_manifest,
    **base_manifest,
}
manifest_file = output / "resume_manifest.json"
if manifest_file.exists():
    if json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError("Model/evaluation configuration changed. Use a different output directory.")
else:
    if list(records.glob("*.json")) or (output / "eval_info.json").exists():
        raise RuntimeError("Existing results have no matching resume manifest; use a new output directory.")
    atomic_json(manifest_file, manifest)

def record_path(group, task_id):
    return records / f"{group}_{task_id}.json"

def read_record(group, task_id, episodes):
    p = record_path(group, task_id)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if r["task_group"] != group or r["task_id"] != task_id:
        raise RuntimeError(f"Invalid task identity in {p}")
    metrics = r["metrics"]
    if (len(metrics.get("successes", [])) != episodes
            or any(type(v) is not bool for v in metrics["successes"])):
        raise RuntimeError(f"Incomplete or invalid saved results: {p}")
    if attention_manifest:
        require_saved_attention_videos(metrics)
    return metrics

state = {"bar": None, "task_done": 0, "task_total": 0,
         "current": "", "task_episodes": 0, "per_task": 0, "successes": 0}

def refresh():
    bar = state["bar"]
    if bar is not None and not progress_fd:
        bar.set_postfix_str(
            f"剩余={int(bar.total - bar.n)} "
            f"任务={state['task_done']}/{state['task_total']} "
            f"当前={state['current']} "
            f"本任务={state['task_episodes']}/{state['per_task']} "
            f"成功={state['successes']}/{int(bar.n)}",
            refresh=True,
        )

def rollout_with_progress(*args, **kwargs):
    result = original_rollout(*args, **kwargs)
    bar = state["bar"]
    if bar is not None:
        # batch_size=1 in this launcher: one returned rollout is one complete episode.
        batch = int(result["done"].shape[0])
        remaining = state["per_task"] - state["task_episodes"]
        accepted = min(batch, remaining)
        # Match the evaluator's success masking rather than reading intermediate progress.
        for i in range(accepted):
            done_index = int(result["done"][i].to(int).argmax().item())
            success = bool(result["success"][i, :done_index + 2].any().item())
            state["successes"] += int(success)
        state["task_episodes"] += accepted
        bar.update(accepted)
        refresh()
    return result

def one_with_progress(*args, **kwargs):
    values = one_signature.bind(*args, **kwargs).arguments
    group, task_id = values["task_group"], int(values["task_id"])
    saved = read_record(group, task_id, int(values["n_episodes"]))
    if saved is not None:
        return group, task_id, saved
    state["current"] = f"{group}/{task_id}"
    state["task_episodes"] = 0
    refresh()
    result = original_one(*args, **kwargs)
    tg, tid, metrics = result
    if tg != group or tid != task_id or len(metrics.get("successes", [])) != state["per_task"]:
        raise RuntimeError("Unexpected/incomplete task result; not marking task completed")
    atomic_json(record_path(group, task_id),
                {"task_group": group, "task_id": task_id, "metrics": metrics})
    state["task_done"] += 1
    refresh()
    return result

def all_with_progress(*args, **kwargs):
    values = all_signature.bind(*args, **kwargs)
    values.apply_defaults()
    values = values.arguments
    if values.get("max_parallel_tasks", 1) != 1:
        raise RuntimeError("This progress wrapper requires max_parallel_tasks=1")
    envs = values["envs"]
    state["task_total"] = sum(len(tasks) for tasks in envs.values())
    state["per_task"] = int(values["n_episodes"])
    state["task_done"] = state["task_episodes"] = state["successes"] = 0
    saved_episodes = 0
    for group, tasks in envs.items():
        for task_id in tasks:
            saved = read_record(group, int(task_id), state["per_task"])
            if saved is not None:
                state["task_done"] += 1
                saved_episodes += len(saved["successes"])
                state["successes"] += sum(saved["successes"])
    print(f"恢复进度：已保存 {state['task_done']} 个任务 / {saved_episodes} 个 episode", flush=True)
    state["current"] = "准备开始"
    progress_options = {}
    if progress_fd:
        progress_options = {
            "desc": os.environ.get("LIBERO_PROGRESS_LABEL", "LIBERO"),
            "file": progress_stream, "leave": False, "ascii": True,
            "mininterval": 1.0,
            "bar_format": "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed} ETA {remaining}]",
        }
    state["bar"] = tqdm(
        total=state["task_total"] * state["per_task"],
        initial=saved_episodes,
        unit="episode", dynamic_ncols=True, disable=False, position=0,
        **({
            "desc": "LIBERO 总进度", "mininterval": 0.2,
            "bar_format": "{desc}: {percentage:5.1f}%|{bar}| 已完成 {n_fmt}/{total_fmt} "
                          "[已用 {elapsed}, 预计剩余 {remaining}, {rate_fmt}] {postfix}",
        } | progress_options),
    )
    refresh()
    try:
        with logging_redirect_tqdm():
            return original_all(*args, **kwargs)
    finally:
        # Durable summary excludes episodes in a task that failed before saving.
        persisted = []
        per_suite = {}
        for group, tasks in envs.items():
            suite_successes = []
            for task_id in tasks:
                metrics = read_record(group, int(task_id), state["per_task"])
                if metrics is not None:
                    persisted.extend(metrics["successes"])
                    suite_successes.extend(metrics["successes"])
            per_suite[group] = {
                "saved_episodes": len(suite_successes),
                "expected_episodes": len(tasks) * state["per_task"],
                "successes": sum(suite_successes),
                "pc_success_saved": 100 * sum(suite_successes) / len(suite_successes) if suite_successes else None,
            }
        atomic_json(output / "resume_summary.json", {
            "per_suite": per_suite,
            "saved_episodes": len(persisted),
            "expected_episodes": state["task_total"] * state["per_task"],
            "successes": sum(persisted),
            "pc_success_saved": 100 * sum(persisted) / len(persisted) if persisted else None,
        })
        state["bar"].close()
        state["bar"] = None

# Hide the evaluator's per-step/per-batch bars so they do not obscure the overall bar.
for name in ("tqdm", "trange"):
    old = getattr(module, name, None)
    if callable(old):
        def quiet_bar(*args, _original=old, **kwargs):
            kwargs["disable"] = True
            return _original(*args, **kwargs)
        setattr(module, name, quiet_bar)
module.rollout = rollout_with_progress
module.run_one = one_with_progress
module.eval_policy_all = all_with_progress
sys.argv[0] = "lerobot-eval"
entry.load()()
PY
  EVAL_EXIT=${PIPESTATUS[0]}
  set -e
  echo "===== eval end $(date) checkpoint=$STEP_PADDED exit=$EVAL_EXIT =====" | tee -a "$LOG"
  if [[ "$EVAL_EXIT" -ne 0 ]]; then
    echo "$STEP_PADDED: evaluation failed (exit=$EVAL_EXIT)" | tee -a "$SUMMARY_LOG"
    FINAL_EXIT=1
    continue
  fi

  EVAL_INFO="$OUTPUT_DIR/eval_info.json"
  if [[ ! -f "$EVAL_INFO" ]]; then
    echo "WARNING: $EVAL_INFO not found" | tee -a "$LOG" "$SUMMARY_LOG"
    FINAL_EXIT=1
    continue
  fi
  python - "$EVAL_INFO" "$STEP_PADDED" <<'PY' | tee -a "$LOG" "$SUMMARY_LOG"
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
tasks = data.get("per_task") or data.get("Aggregated Metrics for per_task")
overall = data.get("overall") or data.get("Aggregated Metrics for overall") or data
if isinstance(tasks, list):
    successes = []
    by_suite = {}
    print("===== per-task success =====")
    for task in tasks:
        metrics = task.get("metrics", task)
        values = metrics.get("successes", [])
        successes.extend(values)
        by_suite.setdefault(task.get("task_group", "unknown"), []).extend(values)
        rate = 100 * sum(bool(v) for v in values) / len(values) if values else float("nan")
        print(f"{task.get('task_group', '?')}_{task.get('task_id', '?')}: "
              f"{rate:.1f}% ({sum(bool(v) for v in values)}/{len(values)})")
    print("===== per-suite success =====")
    for suite, values in sorted(by_suite.items()):
        rate = 100 * sum(bool(v) for v in values) / len(values) if values else float("nan")
        print(f"{suite}: {rate:.2f}% ({sum(bool(v) for v in values)}/{len(values)})")
    if successes:
        print(f"===== checkpoint {sys.argv[2]} overall success rate: "
              f"{100 * sum(bool(v) for v in successes) / len(successes):.2f}% "
              f"(n_episodes={len(successes)}) =====")
    else:
        print("Overall:", overall)
else:
    print("Overall:", overall)
PY
done
if [[ "$DRY_RUN" == true ]]; then
  exit "$FINAL_EXIT"
fi
echo "================ summary ================"
cat "$SUMMARY_LOG"
exit "$FINAL_EXIT"
