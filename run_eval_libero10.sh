#!/bin/bash
set -e
# Activate your existing lerobot environment before running.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset MUJOCO_EGL_DEVICE_ID || true
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/wyn/.libero}"
export PYTHONUNBUFFERED=1

BASE_CKPT=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-direct_dual-seed0/checkpoints
# BASE_CKPT=/root/autodl-tmp/chkpt/2601-lerobot/912/checkpoints
BASE_OUTPUT=/root/autodl-tmp/eval/2601-lerobot
STEPS=(3000)
mkdir -p "$BASE_OUTPUT"
# SUMMARY_LOG="$BASE_OUTPUT/eval_pi05-dual_prompt_only-seed0-summary.log"
SUMMARY_LOG="$BASE_OUTPUT/eval_pi05-direct_dual-seed0-summary.log"
touch "$SUMMARY_LOG"
FINAL_EXIT=0

for STEP in "${STEPS[@]}"; do
  STEP_PADDED=$(printf "%06d" "$STEP")
  CKPT="$BASE_CKPT/$STEP_PADDED/pretrained_model"
  OUTPUT_DIR="$BASE_OUTPUT/libero060-all-pi05-direct_dual-seed0-${STEP_PADDED}-libero10-resume"
  LOG="$BASE_OUTPUT/eval_pi05-direct_dual-seed0-${STEP_PADDED}.log"
  # OUTPUT_DIR="$BASE_OUTPUT/libero060-all-pi05-912-1-${STEP_PADDED}-libero10-resume"
  # LOG="$BASE_OUTPUT/eval_pi05-912-1-${STEP_PADDED}.log"
  touch "$LOG"
  {
    echo "===== evaluating checkpoint $STEP_PADDED at $(date) ====="
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "policy=$CKPT"
    echo "output_dir=$OUTPUT_DIR"
    echo "log=$LOG"
  } | tee -a "$LOG"

  if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" | tee -a "$LOG" "$SUMMARY_LOG"
    FINAL_EXIT=1
    continue
  fi

  set +e
  # Pass arguments normally; stdin contains only the progress wrapper.
  python -u - \
    --policy.path="$CKPT" \
    --output_dir="$OUTPUT_DIR" \
    --env.type=libero \
    --env.task=libero_10 \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
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

entries = list(importlib.metadata.entry_points(group="console_scripts", name="lerobot-eval"))
if len(entries) != 1:
    raise RuntimeError("Cannot uniquely resolve lerobot-eval in this Python environment")
entry = entries[0]
module = importlib.import_module(entry.module)
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
    return metrics

state = {"bar": None, "task_done": 0, "task_total": 0,
         "current": "", "task_episodes": 0, "per_task": 0, "successes": 0}

def refresh():
    bar = state["bar"]
    if bar is not None:
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
    state["bar"] = tqdm(
        total=state["task_total"] * state["per_task"],
        initial=saved_episodes,
        desc="LIBERO 总进度", unit="episode", dynamic_ncols=True,
        disable=False, mininterval=0.2, position=0,
        bar_format="{desc}: {percentage:5.1f}%|{bar}| 已完成 {n_fmt}/{total_fmt} "
                   "[已用 {elapsed}, 预计剩余 {remaining}, {rate_fmt}] {postfix}",
    )
    refresh()
    try:
        with logging_redirect_tqdm():
            return original_all(*args, **kwargs)
    finally:
        # Durable summary excludes episodes in a task that failed before saving.
        persisted = []
        for group, tasks in envs.items():
            for task_id in tasks:
                metrics = read_record(group, int(task_id), state["per_task"])
                if metrics is not None:
                    persisted.extend(metrics["successes"])
        atomic_json(output / "resume_summary.json", {
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
    print("===== per-task success =====")
    for task in tasks:
        metrics = task.get("metrics", task)
        values = metrics.get("successes", [])
        successes.extend(values)
        rate = 100 * sum(bool(v) for v in values) / len(values) if values else float("nan")
        print(f"{task.get('task_group', '?')}_{task.get('task_id', '?')}: "
              f"{rate:.1f}% ({sum(bool(v) for v in values)}/{len(values)})")
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
echo "================ summary ================"
cat "$SUMMARY_LOG"
exit "$FINAL_EXIT"
