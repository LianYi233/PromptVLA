# Deterministic tabletop-object position shifts for task7

This is an **evaluation-only** perturbation of the existing 1200-step,
single-episode task7 adaptation experiment. It does not retrain the models.

## Physical perturbation and fairness

Task7's standard instruction is "put both the alphabet soup and the cream
cheese box in the basket". The official task BDDL names two target objects
`alphabet_soup_1` and `cream_cheese_1`, and the receptacle
`basket_1`. Only the two targets are moved in the tabletop X/Y plane
after loading the saved official LIBERO initial state and **before** the
normal settling steps. The basket, ketchup, tomato sauce, robot pose,
object heights and rotations, reward definition, instruction, and model
weights are never changed. The code requires a single 7-DOF free joint
per object and rejects physically crowded placements rather than
silently falling back to a different shift strength.

At a given displacement magnitude, every target is displaced by that
magnitude (randomized direction, exact radius) and placed at least 7 cm
from each other object's center. Maximum allowed displacement 14 cm; the target centres must also remain within a conservative 5 cm margin from the living-room tabletop edges.
This is a *heuristic* geometry check, not a formal collision-free placement
guarantee; visually inspect a few perturbed rollouts before drawing
conclusions. Perturbations use a SHA256→NumPy PCG64 stream derived solely
from the fixed shift seed, task, rollout seed, target object and strength.
The evaluated model or GPU is never part of this key.

The evaluator also locks **saved initial-state index** to the episode
number derived from the rollout seed, independently of any success
autoresets. It resets the policy action-generation RNG per episode to
a shared seed. Consequently all four models receive the same task7
initial states and *intended* object displacements, even if the model
trajectories diverge. Exact end-to-end physics replay still depends on
the simulation/GPU software stack; keep that environment fixed.

No image noise is used. Standard LIBERO-long `libero_10` task7 alone is
evaluated, 10 episodes per model.

## 1. Update code (do not delete existing 1200-step models)

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/task7-fixed-object-shift-strong
git switch -c feat/task7-fixed-object-shift-strong FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

If already on this branch, use `git merge --ff-only FETCH_HEAD`
after fetching.

## 2. First calibrate with Uniform ONLY, independent pilot episodes

The first object-shift pilot achieved 9/10 at 2 cm, 7/10 at 4 cm, and
8/10 at 6 cm. The closest was 70%, above the desired 50%.
This follow-up tests stronger perturbations (8, 10, 12, 14 cm) and saves
a NEW record, leaving the earlier pilot untouched.
The original Uniform and Event each achieved 9/10 clean task7 success.
A physically plausible shift that gets **close** to 50% cannot be
guaranteed in advance. Calibrate on **Uniform only**, not by peeking at
the four final models.

```bash
GPU_ID=1 python -m lerobot.scripts.pi05_task7_shift_calibrate calibrate \
  --gpu 1 --repeat 1 --episodes 10 \
  --pilot-seed 2000 --final-seed 1000 --shift-seed 20261103 \
  --strengths 0.08,0.10,0.12,0.14
```

This evaluates the **same saved Uniform checkpoint** at 8, 10, 12, 14 cm,
10 pilot episodes each, and writes the full decision and success counts
to:

```
/data1/wyn/prompt-work/gripper-event-task7-1200/shift-calibration-strong.json
```

It picks the strength with pilot Uniform success nearest 50% (ties go to
smaller shift). With 10 rollouts, successes change in 10-point increments.
If all candidates remain far from 50%, the report will say so; do NOT
claim the final benchmark is a 50%-difficulty task or automatically
scale to implausibly large object translations. The pilot seed is 2000;
the final test seed is 1000, so the final positions and policy
randomness are not the pilot's exact realization.

To manually choose 10 cm without calibration (if the pilot is too
expensive), pass `--shift-m 0.10` in the final command. The final
baseline rate must then be measured, not assumed to be 50%.

## 3. Evaluate four models in EXACTLY two simultaneous pairs

```bash
DRY_RUN=true GPU_IDS=1,3 \
  bash run_eval_libero_gripper_task7_shift_2gpu.sh \
  --repeat 1 --eval-seed 1000 --episodes 10

GPU_IDS=1,3 \
  bash run_eval_libero_gripper_task7_shift_2gpu.sh \
  --repeat 1 --eval-seed 1000 --episodes 10
```

If `--shift-m` is omitted, the two-GPU runner automatically reads the
calibrated magnitude from the saved record. If calibration has not
been run, you can explicitly pass e.g. `--shift-m 0.10`.
For both cases the shift seed defaults to 20261103; override with
`--shift-seed` only if necessary, but keep the same value in every run.

**Phase 1 (both GPUs at once):** GPU 1 = Uniform, GPU 3 = Event.
After both complete:
**Phase 2 (both GPUs at once):** GPU 1 = Uniform+Time,
GPU 3 = Event+Time.

The runner waits for phase 1 before phase 2 and refuses to create a
combined report unless all four results are valid. Each case runs
10 standard task7 episodes (no other tasks), with **no image noise**.

Models are the original `pi05-gripper-additive-r1-<case>-task7-epN-
base1000-extra200-steps1200-...` checkpoints at
`checkpoints/001200/pretrained_model`; no new training is needed.

You can also evaluate individual cases with:

```bash
GPU_ID=1 bash run_eval_libero_gripper_task7_shift.sh uniform \
  --repeat 1 --eval-seed 1000 --episodes 10 \
  --shift-m 0.10 --shift-seed 20261103
```

Results are segregated from the clean results under:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-task7-object-shift/fixed-object-shift/
```

Every strength/seed produces a separate resumable task7-only directory.
The MuJoCo displacement protocol, source code SHA and episode count
are included in the saved resume manifest and checked when summarizing.

Logs are under:

```
/data1/wyn/chkpt/2601-lerobot/logs/gripper-task7-object-shift/
```

For reproducibility checks:

```bash
python -m pytest -q tests/scripts/test_pi05_task7_object_shift.py
```

## Scientific caution

Do not adjust the shift strength based on final Event or Time results,
or re-select it after seeing the final test seed. A baseline of 50%
cannot be guaranteed from the position-shift magnitude alone. The
fixed protocol yields a fairer comparison, but 10 episodes is only a
quick exploratory test (one additional success moves the rate by 10 pp).
For publication, repeat on more held-out starts and independent
training seeds.
