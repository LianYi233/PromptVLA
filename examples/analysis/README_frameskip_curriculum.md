# FrameSkip-inspired PromptVLA curricula

This is an adaptation of [FrameSkip](https://arxiv.org/abs/2605.13757), not an exact reproduction. It tests **interaction importance and its scheduling**, not a proven easy-to-hard difficulty measure. Frozen PI05 SigLIP features replace DINOv2, and a gripper-transition neighborhood replaces annotated task-progress priors. No changes to the PI05 model, flow objective, original training loop, or processors are required.

## Fixed recipe and names

- `pi05_libero_base`, frozen backbone, VLM-only token=2, action prompts=0, imported local P1, D0 task slots, global seed0.
- 2 GPUs/run, batch32 **per GPU** (global64), 3000 updates; original LR/warmup (1000 optimizer warmup steps), bf16, compile disabled.
- The **data warmup is 500 updates**; it is distinct from the original optimizer's 1000-step LR warmup.
- Full model saved **only at 003000**. Existing small prompt/rank probes and training/input diagnostics remain enabled.
- Two joint noise/time banks: E0/T0 and E1/T1. All draws follow immutable sample-use IDs (`Bocc`). This is not a separate E-vs-T main-effect experiment.
- Example name: `pi05-fskip-r1-C-Bocc-base-vlm2-P1-D0-E1-T1`.
- `r1` is an experiment label; changing it does **not** change any stochastic seed.

| Label | First 500 updates | Updates 501–3000 |
|---|---|---|
| U | Same full-frame sample uses and ε/t within each bank | Uniform full-frame sampling within the fixed task slots |
| R | Same | 50% source draws from a random retained pool, 50% from full frames |
| S | Same | 50% source draws from a scored retained pool, 50% from full frames |
| C | Same | Scored-pool source share increases approximately 20%→80%, averaging exactly 50% |

S/C share **the exact same 160,000 post-warmup sample-use instances**, 80,000 selected-source and 80,000 full-source, including the same ε/t for each use. Their order/batch composition differs, while the task at every global slot is unchanged. Per-task fixed-size randomized rounding preserves the exact shared task budgets; a small intercept correction makes each task's ramp sum to that budget. Realized window fractions are recorded, not assumed equal to the nominal curve.

R has the same per-episode selected-pool size and protected frames as S/C. It tests the score against random selection. U/R/S intentionally differ in sampled content, so **only S/C isolate ordering with matched data/noise pairing**. Sampling is with replacement after warmup. Repeated uses of one frame get separate use IDs/noise; noise is not permanently attached to a frame. The warmup is built from the D0/W0 finite-plan prefix. These are new runs, not replicas of earlier orderbind results.

A pool-source percentage is **not** the actual percentage of keyframes: the full pool also contains selected frames. `schedule_report.csv` reports both. A 20% retained pool is an enrichment source, not a reduction to 20% of the training budget. No sample is removed from the underlying dataset or renumbered; every selected start still loads its original contiguous 50-action chunk with the existing end-of-episode padding/masks. Evaluation executes 10 actions as before.

## Scoring recipe

For each chronological episode:

1. Scale the six continuous action coordinates with training `q01/q99`; exclude the gripper. Action importance is the norm of the adjacent-frame action change plus 0.1 times the mean per-coordinate variance of up to the next three actions. The first change is zero; no difference crosses an episode boundary.
2. Load **only the frozen vision tower** from local base-model safetensors. Use PI05 resizing/padding and [-1,1] input scaling, float32/eager inference, mean-pool last-layer patch features and L2-normalize. Compute adjacent-frame changes independently for both cameras. Divide by max(normalized action-change norm, 0.05), clip each camera's ratio at its episode 95th percentile, and average the two ratios.
3. Decode binary gripper commands (0/1 threshold0.5, or -1/1 threshold0). Require two consecutive new-state commands to confirm a transition. Mark the neighborhood [-10,+2]. Nonbinary gripper data requires an explicit `--gripper-threshold` after schema inspection.
4. Robust episode normalization (1st/99th percentiles, min/max fallback for sparse nonconstant signals; flat signals contribute zero). Score = **0.6 action + 0.2 visual + 0.2 event-neighborhood**. These are starting hyperparameters, not validated optima.
5. Retain ceil(20% × episode length), protecting first/last frames and confirmed transition frames. If protected frames alone exceed the quota, expand the quota. Break score ties with a fixed episode-specific RNG. The random control has the exact same protected frames and final count in every episode.

This is a score of the **chunk start**, not an average of all 50 future steps, and a gripper switch does not prove contact or successful grasp. There is no midpoint Gaussian or manually guessed task-progress prior. Scores, both camera deltas/ratios, event flags, source provenance and both selection masks are stored separately in `scores.npz`. Preview plots cover two deterministic episodes per task; `episodes.csv` records actual retention. Vision deltas are cached per episode and resumed after interruption. Scoring runs offline without consuming training RNG.

## 1. Score once, then prepare once

Use the existing `lerobot` environment in the updated checkout. No new large model download is needed. The environment needs its existing training dependencies plus matplotlib for preview plots.

```bash
cd /home/wyn/VLAA/PromptVLA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Optional short pilot: 2 episodes/task. Inspect preview/ before doing all episodes.
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.frameskip_scores \
  --output /data1/wyn/prompt-work/frameskip-r1 --pilot-only

# Full scoring reuses the pilot's cached visual deltas.
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.frameskip_scores \
  --output /data1/wyn/prompt-work/frameskip-r1

python -m lerobot.scripts.frameskip_experiment prepare \
  --scores /data1/wyn/prompt-work/frameskip-r1/scores.npz \
  --output /data1/wyn/prompt-work/frameskip-r1/plans
```

Defaults: dataset `/home/wyn/datasets/libero`, source `/home/wyn/models/lerobot/pi05_libero_base`. Two cameras are inferred from dataset metadata; override explicitly with `--cameras observation.images.image observation.images.image2`. `--batch-size` only affects offline vision extraction, not training. Scoring does not instantiate the 4B policy/LLM/action expert. All saved vision keys must load strictly; missing weights never fall back to random initialization.

Run scoring/preparation **once before starting the two queues**. Do not concurrently write the same vision cache, modify score code/config, or regenerate plans during a run. A cache source/config change is rejected; use a new scoring directory if deliberately changing the recipe. Plot labels use camera0/camera1; the exact camera-key mapping is recorded in the score manifest. Look for camera shake/near-static denominator artifacts, meaningful action/gripper neighborhoods, and coverage across time. All raw frames remain accessible to training regardless of selection.

## 2. Run both queues

Terminal A:

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29561 \
bash examples/training/train_pi05_frameskip_curriculum.sh a \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

Terminal B:

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29562 \
bash examples/training/train_pi05_frameskip_curriculum.sh b \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

| Round | GPU0,1 (A) | GPU2,3 (B) |
|---|---|---|
| 1 | U, E0/T0 | U, E1/T1 |
| 2 | R, E1/T1 | R, E0/T0 |
| 3 | S, E0/T0 | S, E1/T1 |
| 4 | C, E1/T1 | C, E0/T0 |

Each queue is sequential. Defaults are explicit and reject conflicting inherited variables. `DRY_RUN=true` performs the actual input/plan/init/model preflights and prints commands without training. `CELLS="S0 C1"` selects only those remaining cells in a queue after interruption; no completed/partial directory is silently skipped, deleted or overwritten. The launcher checks all selected outputs before starting the first model.

Full models go to `/data1/wyn/chkpt/2601-lerobot/PromptVLA/`; logs to `/data1/wyn/chkpt/2601-lerobot/logs/`. Scores/plans/cache stay under `/data1/wyn/prompt-work/frameskip-r1/`. The imported initialization is `/data1/wyn/prompt-work/pi05-base-vlm2-init-seed1.safetensors`.

## 3. Audit and evaluate

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.frameskip_experiment audit-runs \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-fskip-r1-*-Bocc-base-vlm2-P1-D0-E*-T*

# Sequential eight-model evaluation, single-line progress plus final rates.
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
GPU_ID=2 PROGRESS=true bash run_eval_libero_frameskip.sh all
```

The evaluator defaults to checkpoint3000, seed1000, 10 episodes/task, all four suites. `CELLS="S0 C0"` restricts evaluation to named cells. Full logs/video paths stay in the existing evaluator's log files. It checks run names, training configs and saved processors before model evaluation.

The final audit checks all step/rank traces, every immutable use ID, per-key CPU noise/time hashes, warmup equality, sample pools, task slots, source-code hashes and the complete four-schedule × two-bank experiment. S/C must have identical sample/noise pairing and different order. Dataset actions/statistics are hashed during scoring and rechecked before launch; videos/full source files use relative-path/size/mtime inventories (not full content hashes). The vision weights themselves have a tensor-content hash. These checks establish the recorded input controls, **not full GPU numerical determinism**.

Compare S−R to test score-informed selection, C−S to test temporal scheduling with matched uses/noise, and R−U to detect random-pool/repetition effects. Report each bank separately and all four suites. Two banks are exploratory evidence; a positive result needs additional independent banks/replicates. Lower flow loss or a larger effective rank alone is not success.

## Local checks

```bash
uv run pytest tests/scripts/test_frameskip_curriculum.py -q
bash -n examples/training/train_pi05_frameskip_curriculum.sh
bash -n run_eval_libero_frameskip.sh
```

CPU tests exercise scoring edge cases, exact quotas, schedule construction, actual Accelerate batch sharding/resume offsets, original-chunk delegation, the actual PDET flow generator and full-trace corruption detection. The real local LIBERO decoding, A100 vision extraction, distributed model training and rollout evaluation must be smoke-tested on the data server; they cannot be validated by the CPU fixtures alone.
