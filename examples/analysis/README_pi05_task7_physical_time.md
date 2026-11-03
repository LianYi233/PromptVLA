# Experiment II · Task7 1000-step Variable Physical Time vs RoPE Token Position

This experiment tests whether explicit physical time provides more information
than a token index once future action TARGETS are sampled at different actual
recorded intervals. It does **not** modify LIBERO simulator control frequency.

## Matched four-way experiment

| Arm | Extra time conditioning | Added trainable parameters |
|---|---|---:|
| `rope` | native pretrained RoPE only | 0 |
| `index` | additive φ(j/fps), ignoring actual Δt | 32,768 |
| `global_dt` | additive φ(Δt) repeated over all 50 action tokens | 32,768 |
| `physical` | additive φ(j·Δt) for every future action token | 32,768 |

All cases keep the original Gemma/Pi0.5 RoPE. The latter three have the same
32-D sinusoidal features (0.1–10 Hz) and a shared-structure zero-initialized
linear projection of dimension 1024×32, injected in the action INPUT pathway.
`global_dt` is a per-example constant across its 50 action positions.
The native RoPE token positions remain the same 0..49 regardless of stride.

All four are initialized from the exact same
`/home/wyn/models/lerobot/pi05_libero_base`, the same imported seed-1
two-token VLM prompt, and the same DDP data/noise/timestep sequence.
The VLM backbone stays frozen and the complete Action Expert + VLM prompt
are trained with batch 32 per GPU, 2 GPUs, exactly 1000 updates per arm.
This model was already pretrained on LIBERO. Do NOT describe as learning
from one demonstration without prior LIBERO exposure.

## Actual data labels (important)

Use the same selected LIBERO-long task7 demo from
`/data1/wyn/prompt-work/gripper-event-task7-1200/base/` (previously episode 7),
but construct a **new** 1000-step Uniform plan from anchors that leave at
least 99 original frames in the episode.

For each matched occurrence, deterministically assign stride k=1 or k=2,
with exactly 50% of the 64,000 occurrences assigned each cadence.

- At k=1, targets are `action[t], action[t+1], ..., action[t+49]`.
- At k=2, targets are `action[t], action[t+2], ..., action[t+98]`.
- Observations/cameras/language are always from the SAME frame t.
- All four models receive the SAME t, k, action labels and flow random
  numbers for every occurrence.

`Δt=k/fps` is computed from the real dataset FPS (10 in the previous
task7 data). `index` deliberately ignores k and always encodes j/fps;
`global_dt` receives the scalar k/fps but no jΔt relation; `physical`
receives the per-position j·k/fps schedule.

Targets are **instantaneous original relative-action commands at those
future times**, not an integrated 2-step control command. **Do not**
execute k=2 or k=3 chunks directly in LIBERO. Correct control-rate
conversion and step compositing are separate experiments. This is an
OFFLINE test of future action-time representation, not a robot-success
evaluation.

All train/heldout start indices are disjoint, but action-target ranges may
overlap within the same trajectory. The heldout metrics are correlated
single-demo diagnostics and do not establish out-of-demonstration generalization.

## Step 1. Update the branch

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/task7-physical-time-cadence-1k
git switch -c feat/task7-physical-time-cadence-1k FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

If the branch already exists locally, `git switch` then
`git merge --ff-only FETCH_HEAD` instead.

## Step 2. Prepare actual strided action labels

```bash
python -m lerobot.scripts.pi05_task7_cadence prepare
python -m lerobot.scripts.pi05_task7_cadence check
```

The new, isolated preparation directory is:

```
/data1/wyn/prompt-work/task7-variable-cadence-1k/
  uniform.npz   # PDET observation/occurrence sample order, 1000 steps
  cadence.npz   # k=1/2, one label per IMMUTABLE occurrence
  manifest.json # dataset/action hashes, episode, FPS, heldout start indices
```

The preparation verifies original data fingerprints and that no target window
escapes the single demonstration. It fails explicitly if the demonstration is
too short for k=2 windows. Heldout starts are every fifth eligible start,
excluded from the sampled training plan.

Optional k=3 (**unseen cadence**) is constructed from heldout anchors only
when 50 targets at t+3j remain inside the same demonstration. If there is
no valid k=3 anchor, its metric is omitted, not fabricated.

## Step 3. Unit tests and 2-GPU DRY_RUN

```bash
python -m pytest -q tests/scripts/test_pi05_task7_physical_time.py

DRY_RUN=true GPU_IDS=0,1 \
  bash examples/training/train_pi05_task7_physical_time.sh all
```

This checks the real native launcher accepts
`--policy.action_time_basis=index|global_dt|physical`, and displays
the exact 1000-step 2-GPU training commands without starting the optimizer.

## Step 4. Train four 1000-step checkpoints

```bash
GPU_IDS=0,1 bash examples/training/train_pi05_task7_physical_time.sh all
```

The four models are trained SEQUENTIALLY using both GPUs for each model.
They are never trained concurrently on a single GPU.

Checkpoints under `/data1/wyn/chkpt/2601-lerobot/PromptVLA/`:

```
pi05-task7-physicaltime-r1-<case>-ep7-steps1000-aexpert-base-vlm2-P1-D0-E1-T1/
  checkpoints/001000/pretrained_model/
```

The saved `cadence_experiment.json`, PDET per-rank sample audit, time basis,
and 1024×32 learned projection are checked. The previous 1200-step task7 and
1000-step QKV-path checkpoints remain separate.

After completion:

```bash
python -m lerobot.scripts.pi05_task7_physical_time_experiment audit all
```

## Step 5. Matched offline evaluation, GPU 0 and GPU 1

```bash
DRY_RUN=true GPU_IDS=0,1 \
  bash run_eval_libero_task7_physical_time_2gpu.sh \
  --eval-seed 3000 --flow-time 0.5 --max-eval-starts 32

GPU_IDS=0,1 \
  bash run_eval_libero_task7_physical_time_2gpu.sh \
  --eval-seed 3000 --flow-time 0.5 --max-eval-starts 32
```

- Phase 1: GPU0 `rope` and GPU1 `index`.
- Phase 2: GPU0 `global_dt` and GPU1 `physical`.
- Batch: one heldout anchor and one cadence per inference.
- For each (anchor, k), all models receive the same normalized action label,
  SHA256-keyed Gaussian Flow noise and fixed Flow time t=0.5.
- Metric: normalized Flow velocity MSE; **lower is better**.
- Report k=1 and k=2 independently; report k=3 as out-of-training cadence
  only if original trajectory length allows valid 50-token labels.

The offline evaluator loads each checkpoint's ORIGINAL normalization,
tokenizer, images and task language. No MuJoCo environment is started and
no action chunk is executed; no success rate is claimed.

Results:
```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-physical-time/
  seed3000-t0.50/
    pi05-task7-physicaltime-r1-<case>-....json
    summary-all.csv
    summary-all.json
```

Logs:
```
/data1/wyn/chkpt/2601-lerobot/logs/task7-physical-time/
```

Different evaluation seeds vary ONLY the fixed Flow noise. They **do not**
create independent visual scenes or episodes; all heldout start frames
remain from one demonstration. Do not interpret repeating eval seeds as
independent simulation trials.

## Interpretation

H1 (physical time adds information): `physical` should improve k=2
relative to the index-only arm and to global Δt, ideally also improve
unseen k=3. A gain over `rope` alone could simply arise from 32,768
extra parameters. `global_dt` is necessary to test whether telling the
model the sample rate alone explains any benefit.

At 10FPS the 0.1–10Hz sine/cosine basis contains aliased high-frequency
components. A further low-frequency ablation is sensible only after these
four cases are understood.

For a paper, repeat on multiple independent original demonstrations,
multiple tasks and independent *training* seeds. This is a controlled
mechanism probe on one pretrained-LIBERO source, not a full VLA benchmark.
