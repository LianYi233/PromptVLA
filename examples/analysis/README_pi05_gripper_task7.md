# LIBERO-long task7: one demonstration, 1200 updates, clean task7-only test

## Scope

Four-condition factorial study (no additional temporal labels):
- **uniform**: 1000 shared uniformly sampled updates + 200 extra uniform
- **event**: the same 1000 shared updates + 200 extra from ±5 frames around
  gripper open→close COMMAND changes
- **uniform_time**: same sample plan as uniform; relative within-action-chunk
  sinusoidal encoding enabled
- **event_time**: same sample plan as event; relative action-time enabled

The **same, first eligible COMPLETE task7 demonstration** is used for
all four conditions; the preparation output records its actual episode index.
Task7 is LIBERO-long suite task ID 7, whose official description is
"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket".
The mixed LeRobot dataset's `task_index` is resolved by exact language and
MUST NOT be assumed equal to suite ID 7. If no episode has a valid detected
gripper closure with both event-near and ordinary 50-step chunk starts, the
preparation fails rather than silently switching datasets or events.

Each group gets exactly 1200 optimizer updates, 2 GPUs × per-GPU batch 32
(76,800 selected chunk starts per condition). The original 1000 updates are
identical in all groups; each receives another 200 updates with equal compute.
All action labels consist of full original contiguous 50-step chunks.

The pretrained source is unchanged: `pi05_libero_base`, VLM prompt P1 (2
tokens), full Action Expert tuning, frozen vision/language backbone. The
relative time embedding uses existing code and is NOT the original global
video frame number. This is continued adaptation from a LIBERO-trained
source, not true one-shot learning from an untrained policy.

**Evaluation is ONLY the standard `libero_10` task7, 10 episodes, 0 image
corruption**, fixed evaluation start seed 1000. Task7 is selected using the
native `LiberoEnv.task_ids=[7]` filter before constructing environments;
no other nine tasks are instantiated or counted. Each result must have exactly
one task7 entry with ten Boolean successes. Summary reports task7 success
and the four factorial differences. It does not report an overall ten-task
success rate; task7-only overall is simply the same task7 rate.


This branch uses a **new work directory** instead of overwriting the previous
800+200 task7 preparation. Four new model names include
`base1000-extra200-steps1200` and end in `checkpoints/001200/pretrained_model`.

## 1. Update the experiment branch

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/gripper-event-task7-1200
git switch -c feat/gripper-event-task7-1200 FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

If already on this branch, update using `git merge --ff-only FETCH_HEAD`
after fetch. Do not delete old task8 checkpoints or sampling plans.

## 2. Prepare the task7 sample and inspect the selection

```bash
python -m lerobot.scripts.pi05_gripper_task7 prepare \
  --dataset-root /home/wyn/datasets/libero \
  --work-dir /data1/wyn/prompt-work/gripper-event-task7-1200 \
  --batch-size 32 --world-size 2
python -m lerobot.scripts.pi05_gripper_task7 check \
  --work-dir /data1/wyn/prompt-work/gripper-event-task7-1200
```

This chooses the first eligible task7 episode from the dataset, recording
the precise episode index and detected closure frames. It does **not** reuse
the old task8 episode10 plan. To select a specific eligible task7 episode,
pass `--episode-index N` when first preparing, and use a new work directory.
The paths are:

- `.../gripper-event-task7-1200/base/uniform.npz`: 1000-step Uniform plan
- `.../gripper-event-task7-1200/additive/uniform.npz`: 1200-step Uniform case
- `.../gripper-event-task7-1200/additive/event.npz`: 1200-step Event case
- `.../gripper-event-task7-1200/additive/manifest.json`: episode, samples, hashes
- `.../gripper-event-task7-1200/additive/anchors.csv`: original/extras counts

## 3. Train four groups sequentially on the same pair of GPUs

```bash
DRY_RUN=true GPU_IDS=1,3 \
  bash examples/training/train_pi05_gripper_task7.sh all

DRY_RUN=false GPU_IDS=1,3 \
  bash examples/training/train_pi05_gripper_task7.sh all
```

Training uses the same 1200 steps, seed and 2×32 batch configuration
in every case. Default W&B mode is offline. Verify both GPUs have adequate
memory before launch (`nvidia-smi`). These four groups train sequentially
with DDP over two GPUs; they are not each on a separate GPU.

Checkpoints under `/data1/wyn/chkpt/2601-lerobot/PromptVLA/` are named
`pi05-gripper-additive-r1-<case>-task7-ep<N>-base1000-extra200-steps1200-aexpert-base-vlm2-P1-D0-E1-T1/checkpoints/001200/pretrained_model`.
These names differ from task8's `003750` checkpoints and will not overwrite
them. `--repeat` changes the run directory label, not the random seed.

## 4. Test: GPU 1 and GPU 3 in parallel, two cases per GPU

```bash
DRY_RUN=true GPU_IDS=1,3 \
  bash run_eval_libero_gripper_task7_2gpu.sh --repeat 1 --episodes 10

DRY_RUN=false GPU_IDS=1,3 \
  bash run_eval_libero_gripper_task7_2gpu.sh --repeat 1 --episodes 10
```

GPU 1 runs `uniform` then `event`; GPU 3 runs `uniform_time` then
`event_time`. The two GPU lanes run concurrently, never loading two
PI05 models on the same GPU at once. Evaluation defaults to seed 1000,
10 episodes/task and strictly no image noise. No `fixed_image_noise_config`
is created. The final checkpoint is `001200`.

Logs: `/data1/wyn/chkpt/2601-lerobot/logs/gripper-task7-1200/<timestamp>-<case>.log`

Results: `/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-task7-1200/`
with task7-only run IDs. The two-GPU script verifies 20,000 MiB free
GPU memory by default before launching (change `MIN_FREE_GPU_MIB`
only if the actual inference footprint is known).

For single-GPU sequential evaluation instead:

```bash
GPU_ID=1 bash run_eval_libero_gripper_task7.sh all \
  --repeat 1 --eval-seed 1000 --episodes 10
```

For a separate report after the four evaluations finish:

```bash
GRIPPER_EVENT_DIR=/data1/wyn/prompt-work/gripper-event-task7-1200/additive \
GRIPPER_EVAL_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-task7-1200 \
python -m lerobot.scripts.pi05_gripper_event_experiment summarize all \
  --only-selected-task --repeat 1 --eval-seed 1000 \
  --episodes 10 --noise-std 0
```

## 5. Limitations

The original time-off/on comparison in task7 was 60% vs 53.3% on the
previous models, but those were trained on task8 and evaluated with noisy
task8-only configuration. New 1200-step task7-only models are a
different protocol; do not treat old scores as their clean baselines.
With only ten rollouts per condition, a single success changes the
estimated rate by ten percentage points. Interpret the first results as a
quick screen, not a statistically conclusive ranking.
