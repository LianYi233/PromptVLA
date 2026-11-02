# Additive Gripper Event Sampling × Relative Time Encoding (LIBERO-long task8)

This is the additive V2 experiment. It is separate from the previous
feat/gripper-event-time-long r1/r2 experiments. The previous sampling plans,
checkpoints and evaluation records are not modified.

## Why change the sampler?

The original Event case had exactly the same number of updates as Uniform:
it replaced a fraction of ordinary samples with gripper-closure samples.
That reduced exposure to approach, transfer and placement phases.

This version NEVER removes an original Uniform occurrence. It reads the exact
saved original uniform.npz and preserves all 3,000 original update batches in
their original relative order. It interleaves **750 new update batches**.
The added batches are uniformly spaced through the training schedule, rather
than concentrated at the end.

**Compute-matched design** (2 GPUs × batch 32, one demonstration, task8 ep10):

| Case | Original uniform steps | Added steps | Total steps | Added start distribution | Relative action-time embedding |
| --- | ---: | ---: | ---: | --- | --- |
| uniform | 3000 | 750 | 3750 | Uniform among legal starts | Off |
| event | 3000 | 750 | 3750 | Within ±5 frames of a gripper-close command | Off |
| uniform_time | 3000 | 750 | 3750 | Uniform among legal starts | On |
| event_time | 3000 | 750 | 3750 | Within ±5 frames of a gripper-close command | On |

Why does Uniform also get 750 extra steps? Without this compute-matched
control, Event would train for 3750 steps and Uniform for only 3000;
a performance change could be caused merely by additional optimization.
All four variants therefore retain exactly the same original 3000 updates
and receive the same additional 750-update budget; only the *extra* examples
and the time-encoding switch change.

All extra steps use the same slot schedule and the same independent random
uniform draw stream. The sample identity changes only in extra slots. PDET
occurrence-bound Gaussian noise and flow timesteps are matched across runs.
Two time-on/off variants reuse the corresponding exact plan files.

**This is NOT globally chronological video playback:** the original Uniform
sequence was random and is preserved as such. Every selected observation
still supervises the following continuous 50-step action target, without
frame skipping or cross-episode joins. Relative action-time embedding is
unchanged from V1 and does NOT encode original global video frame numbers.

## Branch and resources

Branch: feat/gripper-event-additive-long

- Model: /home/wyn/models/lerobot/pi05_libero_base
- Dataset: /home/wyn/datasets/libero
- Tokenizer: /home/wyn/models/paligemma-3b-pt-224
- Initial VLM prompt: existing P1 two-token initial snapshot
- Original base stream: /data1/wyn/prompt-work/gripper-event-long-task8-r1/uniform.npz
- New prepared plan: /data1/wyn/prompt-work/gripper-event-additive-task8-r1
- Output root: /data1/wyn/chkpt/2601-lerobot/PromptVLA
- W&B: offline by default; no online connectivity required

## 1. Fetch and activate branch

    conda activate lerobot
    cd /home/wyn/VLAA/PromptVLA
    git -c core.sshCommand="ssh -p 443" fetch \
      git@ssh.github.com:LianYi233/PromptVLA.git feat/gripper-event-additive-long
    git switch -c feat/gripper-event-additive-long FETCH_HEAD
    python -m pip install --no-deps -e .
    export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

If the branch already exists locally:

    git switch feat/gripper-event-additive-long
    git merge --ff-only FETCH_HEAD

## 2. Preserve the original plan and create the additive plans

    python -m lerobot.scripts.pi05_gripper_additive prepare \
      --dataset-root /home/wyn/datasets/libero \
      --base-plan /data1/wyn/prompt-work/gripper-event-long-task8-r1/uniform.npz \
      --output /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
      --task-id 8 --episode-index 10 \
      --base-steps 3000 --extra-steps 750 \
      --batch-size 32 --world-size 2 \
      --window-before 5 --window-after 5

The source plan must match the dataset, task, episode, dimensions, seed,
and original 3000-update budget. On mismatch the command FAILS rather than
quietly regenerating a different baseline.

Output:
- manifest.json: source provenance, counts, case statistics, exact settings
- uniform.npz: identical 3000 original base batches plus uniform extras
- event.npz: identical 3000 original base batches plus closure-only extras
- anchors.csv: each original anchor's original count, extra counts, totals

Verify before launching:

    python -m lerobot.scripts.pi05_gripper_additive check \
      /data1/wyn/prompt-work/gripper-event-additive-task8-r1

For base 3000, extra 750, per-device batch 32, world size 2:
- Original base draws per group: 192,000 (unchanged)
- Additional draws per group: 48,000
- Total draws per group: 240,000
- Extra near-closure fraction for Event: 100%
- Additional near-closure exposures occur without reducing any original draw

With a different extra budget, use a new --output directory and maintain the
same budget across all four conditions.

## 3. Train four matched variants, offline by default

    DRY_RUN=true GPU_IDS=0,1 \
      bash examples/training/train_pi05_gripper_additive.sh all \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1

    DRY_RUN=false GPU_IDS=0,1 \
      bash examples/training/train_pi05_gripper_additive.sh all \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
        --wandb-mode offline

Each run trains VLM2 + complete Action Expert; the VLM/vision backbone is
frozen. All four cases use the same source, prompt P1, D0/E1/T1 seeds,
base+extra schedule, 3750 optimizer steps and matching learning-rate recipe.
Each saves only checkpoint 003750. Runs are named
pi05-gripper-additive-r1-<case>-task8-ep10-base3000-extra750-steps3750-aexpert-...
No checkpoint from the prior experiment is overwritten.

If only one case is interrupted, rerun that case with a new --repeat label.
For a four-way factorial comparison, all four cases should use the same
repeat label. --repeat is a run identifier, NOT a new random seed.

## 4. Evaluation: same standard LIBERO-long / libero_10

To run 4 GPUs in parallel (one case per GPU), use:

    GPU_IDS=0,1,2,3 \
      bash run_eval_libero_gripper_additive_parallel.sh \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
        --repeat 1

Or evaluate sequentially:

    GPU_ID=0 bash run_eval_libero_gripper_additive.sh all \
      --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 --repeat 1

Evaluation only targets standard libero_10 (10 tasks, default 10 episodes/task).
The parallel entry writes logs under LOG_ROOT/eval-additive-<case>.log.
It summarizes task8, all ten tasks and the four factorial contrasts. To
re-summarize:

    python -m lerobot.scripts.pi05_gripper_event_experiment summarize all \
      --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
      --repeat 1

For more reliable conclusions, consider EPISODES_PER_TASK=100 and multiple
actual TRAINING seeds; adding repeat labels alone does not change randomness.

## Important interpretation

Event vs Uniform is now a **compute-matched extra-exposure comparison**:
both have the exact original uniform training stream and the same additional
budget; only extra sample locations differ. Extra event starts are sampled
from the original closure neighborhood. Original action labels remain
continuous. The observation/action pairs are never concatenated across
randomly drawn start times.

This is different from the first 3000-step Event experiment (where event
overweighting replaced ordinary training starts). Do not pool those two
experiments as though they were the same intervention.
