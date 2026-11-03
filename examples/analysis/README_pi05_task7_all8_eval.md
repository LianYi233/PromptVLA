# Task7 unified 8-model evaluation (GPU 0 + GPU 1)

This launcher combines **both** existing 1000-step studies. It is
not a new training recipe; it requires all eight separately trained
`001000/pretrained_model` checkpoints.

- **Experiment I (4 models):** `rope`, `time_qk`, `time_v`,
  `time_input`. Evaluate actual LIBERO-long task7 rollouts with fixed
  12 cm shifts and standard control rate. Each seed has 10 episodes.
  Report per-seed **success rate** and pooled successes/episodes.
- **Experiment II (4 models):** `rope`, `index`, `global_dt`,
  `physical`. Offline test recorded instantaneous action targets at
  1×/2× (optional 3×) spacing, with fixed Flow time/noise. Report
  **normalised Flow velocity MSE**, lower is better. This is not a
  success-rate evaluation, and 2×/3× commands are NEVER executed.

These are different policies (separate model training datasets) and
different metrics. The two `rope` arms are **not** the same checkpoint,
and the numerical scores **must not** be placed in one pooled ranking.

## 1. Get the code

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/task7-physical-time-cadence-1k
git switch feat/task7-physical-time-cadence-1k
git merge --ff-only FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. Test the unified report logic (no GPU)

```bash
python -m pytest -q tests/scripts/test_pi05_task7_joint_eval.py

DRY_RUN=true GPU_IDS=0,1 bash run_eval_libero_task7_all8_2gpu.sh \
  --timepath-seeds 3000
```

The dry-run prints the two existing underlying double-GPU commands.
It does not load datasets/checkpoints or start any simulation.

## 3. Run all eight checkpoints; quick version

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_all8_2gpu.sh \
  --timepath-seeds 3000 \
  --cadence-seed 3000 \
  --shift-m 0.12 --shift-seed 20261103 \
  --flow-time 0.5 --max-eval-starts 32
```

This runs 4 + 4 models, using at most 2 GPUs concurrently.
Experiment I phases: GPU0 `rope` / GPU1 `time_qk`;
then GPU0 `time_v` / GPU1 `time_input`.
Experiment II phases: GPU0 `rope` / GPU1 `index`;
then GPU0 `global_dt` / GPU1 `physical`.

## 4. Multi-seed experiment I + one-seed experiment II (recommended)

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_all8_2gpu.sh \
  --timepath-seeds 1000,3000,4000,5000 \
  --cadence-seed 3000 \
  --shift-m 0.12 --shift-seed 20261103 \
  --flow-time 0.5 --max-eval-starts 32
```

These defaults intentionally exclude shift calibration seed 2000. The
4×10 eval episodes per experiment-I model use matched initial states
and environment displacement configurations across methods. Different
eval seeds can reuse the same official starting-state indexes.
Experiment-II eval seed changes the fixed Flow noise only; it does
not produce new trajectories or independent held-out demonstrations.

For additional options:
```bash
python -m lerobot.scripts.pi05_task7_joint_eval --help
```

The combined terminal report is printed AFTER all required jobs
finish. Example structure (NOT real results):

```text
[Experiment I] Q/K/V/Input injection · LIBERO-long task7 · 12 cm object shift
Eval seed   RoPE             Time-QK          Time-V           Time-Input
1000        ...              ...              ...              ...
3000        ...              ...              ...              ...
4000        ...              ...              ...              ...
5000        ...              ...              ...              ...
POOLED      successes/40     successes/40     successes/40     successes/40
Time-Input − RoPE: ... pp; Time-Input − Time-QK: ... pp

[Experiment II] Index vs physical time · OFFLINE flow-velocity MSE (↓ better)
Cadence     Starts    RoPE       Index      Global-Δt     Physical
1x          ...       ...        ...        ...           ...
2x          ...       ...        ...        ...           ...
3x          ...       ...        ...        ...           ...  # optional
Physical vs Index: ... % MSE; vs Global-Δt: ... % MSE
```

A preflight checks for all eight checkpoints before running.
If a complete `summary-all.json` already exists, it is reused only
after checking raw `eval_info.json` + displacement manifests
(experiment I), or model fingerprint + heldout cadence manifest
(experiment II). Partial completed evaluations resume using the
original per-experiment runner. Errors remain fatal rather than
printing apparently complete tables with missing cases.

## 5. Print the final tables again WITHOUT running the GPUs

```bash
python -m lerobot.scripts.pi05_task7_joint_eval report \
  --timepath-seeds 1000,3000,4000,5000 \
  --cadence-seed 3000
```

You can evaluate/report one study at a time with
`--only timepath` or `--only cadence`.

The original per-experiment output directories remain intact:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-timepath-qkv/
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-physical-time/
```

The unified report also writes three distinct files under:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-eightway-summary/
  eightway-<config-hash>.json
  eightway-<config-hash>-timepath.csv
  eightway-<config-hash>-cadence.csv
```

The config hash prevents changing the selected seeds/shift/Flow time
from overwriting a different unified comparison.

## 6. Training provenance and limitations

- Both experiments train each model for exactly **1000 updates**,
  but use different action target constructions. Do not reuse an
  experiment-I model to stand in for experiment-II's `rope` baseline.
- Each experiment uses a **single** task7 demonstration and an
  already LIBERO-pretrained source. The studies are controlled
  mechanism probes, not independent few-shot generalization results.
- Experiment-II held-out start frames are disjoint from sampled
  training anchors, but target windows may overlap within the same
  demonstration. A 3× cadence requires valid data (and is optional).
