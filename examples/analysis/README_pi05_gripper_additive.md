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

## 4. Evaluation: deterministic Gaussian RGB perturbation

This experiment now defaults to a **fixed, model-independent image-noise
protocol** when using the `run_eval_libero_gripper_additive*.sh` launchers:

- Default noise: additive independent-pixel Gaussian, sigma=0.10 on images
  normalized to [0,1], with clipping to [0,1].
- Default scope: task8 ONLY (both agentview and wrist cameras). The other
  nine LIBERO-long tasks remain clean. The overall 10-task score therefore
  combines 1 noisy task + 9 clean tasks; use `--noise-scope all` for a
  uniformly perturbed ten-task benchmark.
- Default noise seed: 20261102; evaluation start seed: 1000.
- Noise is generated independently for every *suite, task, episode seed,
  observation step, camera*, with SHA256-derived NumPy PCG64 seeds. It does
  not depend on the case, model weights, GPU ID, or order of completed tasks.
- The same noise realization is therefore used at the same episode/time
  index/camera for all four models. Different model actions can produce
  different clean scene images; common noise does not make trajectories equal.
- The model's torch random stream is also seeded per episode using an
  independent deterministic key, **including when sigma=0 in this launcher**.
  This improves comparability when task durations differ or task-level
  resume skips finished tasks. Physics may still exhibit
  hardware-specific nondeterminism.
- Corruption is applied **after** the LIBERO flip, **before** the saved policy
  preprocessing. Robot states, task language, actions, simulation state and
  checkpoint parameters are not changed. No train-time noise is added.
- The JSON protocol is saved under the separate noisy-evaluation directory
  as `fixed_image_noise_config.json`. All four rollout resume manifests
  contain its SHA256 and the corruption helper code hash. A changed protocol
  cannot resume into old results.

Run four models in parallel (one checkpoint per GPU) with explicit settings:

    DRY_RUN=true GPU_IDS=0,1,2,3 \
      bash run_eval_libero_gripper_additive_parallel.sh \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
        --repeat 1 --eval-seed 1000 --episodes 30 \
        --noise-std 0.10 --noise-seed 20261102 --noise-scope task8

    DRY_RUN=false GPU_IDS=0,1,2,3 \
      bash run_eval_libero_gripper_additive_parallel.sh \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
        --repeat 1 --eval-seed 1000 --episodes 30 \
        --noise-std 0.10 --noise-seed 20261102 --noise-scope task8

`--episodes 30` means 30 initial states per task, or 300 rollouts per
model, and 1200 across all four models. Use `--episodes 10` for a faster
initial check or `--episodes 100` for a more reliable comparison.

To add the exact same noise to all ten standard tasks instead:

    GPU_IDS=0,1,2,3 \
      bash run_eval_libero_gripper_additive_parallel.sh \
        --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
        --repeat 1 --eval-seed 1000 --episodes 30 \
        --noise-std 0.10 --noise-seed 20261102 --noise-scope all

Increase to `--noise-std 0.15` for a stronger corruption (this does not
guarantee a particular success-rate target). Use `--noise-std 0` with this
additive launcher for a **matched clean control**: pixels are unmodified,
but the same per-episode policy RNG rule is retained. This is preferable
when attributing a clean-vs-noisy performance difference to vision noise.
The old historical clean evaluation (before this change) used a different
policy RNG schedule and should not be directly treated as this matched
clean control.

For an apples-to-apples task8 comparison with 30 episodes, run the same
command twice, first with `--noise-std 0`, then with
`--noise-std 0.10`, keeping all other arguments fixed.
Every changed sigma/scope/noise seed has a separate result root, making
accidental reuse of old clean results unlikely. The resume manifest
provides a second strict configuration check.

Single-GPU/sequential evaluation is still available:

    GPU_ID=0 bash run_eval_libero_gripper_additive.sh all \
      --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
      --repeat 1 --noise-std 0.10 --noise-seed 20261102 \
      --noise-scope task8 --episodes 30

Evaluation only supports standard libero_10, not LIBERO-PLUS/PRO. Parallel
logs include a UTC timestamp and per-invocation PID in their names:
`LOG_ROOT/eval-additive-fixednoise-<timestamp>-<pid>-<case>.log`.
The summary reports task8, all ten tasks, four factorial contrasts, and
the effective noise protocol. After partial evaluations, rerun with the
identical arguments to resume. To re-summarize separately:

    python -m lerobot.scripts.pi05_gripper_event_experiment summarize all \
      --directory /data1/wyn/prompt-work/gripper-event-additive-task8-r1 \
      --repeat 1 --episodes 30 --eval-seed 1000 \
      --noise-std 0.10 --noise-seed 20261102 --noise-scope task8

For more reliable conclusions, repeat the same noise protocol across
multiple actual TRAINING seeds; merely changing --repeat does not create a
new training seed. The task8-only mode should not be called full
10-task noise robustness, because nine tasks are unperturbed.

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
