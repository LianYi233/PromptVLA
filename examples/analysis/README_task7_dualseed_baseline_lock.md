# Task7 scene/policy independent RNG · RoPE baseline difficulty lock

## Purpose

The original object-shift protocol used `eval_seed` as BOTH the seed that
affected the direction of 12 cm object displacement and the source for policy
sampling RNG. As a result, 6000 vs 7000 measured **different scene layouts
AND different flow samples**. A single baseline of 20%, 40%, or 70% with
10 episodes is not a fixed model property.

The observed RoPE 6000+7000 pooled result was **9/20=45%**. This is already
within the user's requested 40–50% difficulty range, but that combination
does **not** independently isolate policy noise from scene variability.

This optional **v2 dual-seed protocol** separates three quantities:

| Variable | Purpose | Changed across models? |
|---|---|---|
| `init_seed=3000` | Gym reset seed 3000..3009 and official saved init states 0..9 | No |
| `scene_seed` | Deterministic XY displacement directions for alphabet soup and cream cheese, each exactly `shift_m` | **No**, for a given scene batch |
| `policy_seed` | Deterministic Torch RNG for Pi0.5 Flow sampling on each episode | **No**, for a matched model comparison; yes across repeated policy RNG draws |

Both the scene RNG and the policy RNG are keyed by **episode slot**
0..9. The static 12 cm displacement distance and original 10 official
init states are shared across all arms. Only the two target objects move.
The original `pi05_task7_object_shift.py` remains **unchanged** and the
new protocol has a different identifier, manifest field, helper and result
folder. Earlier 6000/7000 results are NOT numerically the same as v2.

## Stage 1. Calibrate RoPE (training unchanged)

The user wants RoPE around **40–50%** before testing interventions. This
is supported as a *calibrated task-difficulty stress test*, NOT a
post-hoc baseline guarantee or a claim that RoPE's true performance
must equal a chosen percentage.

Predeclare a fixed scene pool (default 20261103,20261104), calibration
policy seed pool (default 101,102), and disjoint heldout test policy
seeds (default 1001,1002,1003). Run **only RoPE**, on the **entire**
predeclared scene×calibration-policy Cartesian product, using 10
episodes per cell. No picking particular high/low-success cells to
make the desired baseline. Only once the POOLED calibration RoPE rate
lies in [40%, 50%] does the program save an immutable lock.

If the observed rate falls outside the band, the script writes a
calibration report, returns a clear failure, and **does not lock**.
It never modifies successes, invents new weights, or silently chooses
favorable seeds. You can explicitly vary task difficulty (for example,
change displacement magnitude, still within 0.14m) **during calibration**,
recording every tried setting and using heldout policy seeds only
AFTER locking. If multiple magnitude candidates are tried, disclose
that model-based difficulty selection in the paper.

### 1. Update the isolated branch

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git \
  feat/task7-dualseed-baseline-lock
git switch -c feat/task7-dualseed-baseline-lock FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

When the branch already exists, use `git switch` then
`git merge --ff-only FETCH_HEAD`.

### 2. Test without launching simulation

```bash
python -m pytest -q tests/scripts/test_pi05_task7_dualseed.py
python -m pytest -q tests/scripts/test_pi05_task7_object_shift.py
python -m pytest -q tests/scripts/test_pi05_task7_input_spectrum.py

GPU_IDS=0,1 bash run_eval_libero_task7_dualseed_2gpu.sh calibrate \
  --dry-run \
  --scene-seeds 20261103,20261104 \
  --cal-policy-seeds 101,102 \
  --test-policy-seeds 1001,1002,1003
```

### 3. Calibrate on GPU 0 and 1

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_dualseed_2gpu.sh calibrate \
  --scene-seeds 20261103,20261104 \
  --cal-policy-seeds 101,102 \
  --test-policy-seeds 1001,1002,1003 \
  --shift-m 0.12 --target-min 40 --target-max 50
```

This runs **2 scenes × 2 calibration policy seeds × 10 episodes = 40**
RoPE-only episodes, two GPU workers concurrently. It prints every
cell and the overall baseline calibration percentage. Successful
calibration creates the immutable file:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-dualseed-locked/
  calibration-lock.json
```

If it fails, there is no lock and Stage 2 refuses to start. The full
calibration report (including unsuccessful trials) remains saved as a
`calibration-<config-hash>.json` file for transparency.

**Do not treat these 40 calibration episodes as heldout evidence.**

## Stage 2. Compare heldout policy seeds on locked scenes

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_dualseed_2gpu.sh compare \
  --scene-seeds 20261103,20261104 \
  --cal-policy-seeds 101,102 \
  --test-policy-seeds 1001,1002,1003 \
  --shift-m 0.12
```

Now evaluate five matched, separately trained 1000-update policies:

1. `rope` (original trained control)
2. `time_input`
3. `lowfreq`
4. `rms_gate`
5. `lowfreq_rms`

Models see the same initial-state IDs, exact object positions and
episode-indexed Flow sampling RNG for every (scene_seed,policy_seed)
cell. The heldout policy seed set does not overlap calibration seeds.

`2 scene seeds × 3 heldout policy seeds × 10 episodes = 60`
independently generated policy-noise test runs per model, under the
two *calibrated* scene layouts (one layout per init state per scene seed).
The script prints per-cell success counts, pooled success rates,
model-only success wins and RoPE-only wins, and saves JSON/CSV.

To also test original Time-QK and Time-V, append:

```bash
  --cases all
```

This makes seven model arms, still max two GPU processes at once.

**The Stage-2 baseline may NOT land at 40–50%.** The locking criterion
only applies to Stage 1. Re-selecting seeds or positions to force
Stage-2 RoPE into the range would bias the comparison. If the Stage-2
baseline moves, report the actual result and investigate variance.

## Recommended stronger generalization check

Evaluate the same models on scenes that were *never involved in the
RoPE difficulty calibration*. Report this separately:

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_dualseed_2gpu.sh compare \
  --scene-seeds 20261103,20261104 \
  --cal-policy-seeds 101,102 \
  --test-policy-seeds 1001,1002,1003 \
  --external-scene-seeds 20261120,20261121 \
  --shift-m 0.12
```

If you want this option, include `--external-scene-seeds
20261120,20261121` **at calibration time as well**, so the complete
comparison plan is recorded in the immutable lock. Otherwise the
comparison rejects the changed plan. These fresh scenes are NOT
used for choosing task difficulty, and their success rates MUST NOT
be mixed into the calibrated-stress summary.

## GPU-free report after evaluations complete

```bash
python -m lerobot.scripts.pi05_task7_dualseed_eval report \
  --scene-seeds 20261103,20261104 \
  --cal-policy-seeds 101,102 \
  --test-policy-seeds 1001,1002,1003 \
  --shift-m 0.12
```

A report verifies the saved task/dataset preparation, baseline
checkpoint fingerprint, all calibration outcomes and each raw
`eval_info.json` plus saved shift/manifests. A missing or changed
manifest is a hard failure: it will not substitute a guessed result.

## Folder structure

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/
  task7-dualseed-locked/
    calibration-lock.json
    calibration-<config-hash>.json
    comparison-calibrated_stress-<config-hash>.json
    comparison-calibrated_stress-<config-hash>.csv
    comparison-unselected_scenes-<config-hash>.json   # optional
    logs/...
  task7-timepath-qkv/
    fixed-object-shift-dualseed/...
  task7-time-input-spectral/
    fixed-object-shift-dualseed/...
```

### Scientific caveats

- The calibrated scene suite is **selected using RoPE's success rate**.
  This is allowed for stress-test design when disclosed, but the target
  cannot be used as evidence that the baseline always obtains 45%.
- Changing `policy_seed` does not create fresh visual layouts; repeated
  policy draws on the same scene are correlated. Thus 60 observations
  are not 60 independent task layouts.
- Models are all adapted from a **LIBERO-pretrained** source using one
  task7 demonstration; the current stress test cannot establish
  low-data generalization to novel tasks.
- For full research conclusions, replicate independent training seeds,
  include new scenes and tasks, and report paired per-episode results
  with confidence intervals.
