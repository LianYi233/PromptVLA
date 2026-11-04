# Task7 · Input-Time Spectral & RMS Study (1000-Step, Two GPUs)

## Why this experiment?

An earlier 1000-update task7 evaluation reported **RoPE 6/10, Time-QK 5/10,
Time-V 5/10, Time-Input 7/10** at *eval seed 5000*. The repeated terminal
lines referred to the **same summary file**, not two independent trials.
One successful episode (10 percentage points) is too little evidence to
claim an improvement. Q/K-only and V-only were first-layer interventions,
whereas Time-Input propagates through the full expert, so the comparisons
do NOT isolate attention content vs attention routing perfectly.

A plausible optimization target is the time-feature spectrum: source
encoding uses 16 frequencies across **0.1–10 Hz**, but the selected task7
data are **10 FPS (Nyquist 5 Hz)**. Frequencies above 5 Hz alias, and a
10-Hz sampled sinusoid degenerates. High-frequency time features could
overfit positional details from one demonstration. This is an *empirical
hypothesis*, not an established explanation for 7/10.

A second hypothesis is that the temporal residual's appropriate magnitude
varies across noisy actions in each 50-step chunk. We test a bounded,
nonparametric RMS-content gate. It has access ONLY to the current
noisy-action embedding, NOT the true future event or task completion time.
It is computed identically during Flow training and all inference steps.

## Four time-on cells (+ one RoPE reference)

| Case | Frequency range | Residual gain | Learned time params | Training |
|---|---|---|---:|---|
| `time_input` | 0.1–10 Hz | constant 1 | 32,768 | **Reuse earlier checkpoint** |
| `lowfreq` | 0.1–4 Hz | constant 1 | 32,768 | **NEW 1000 updates** |
| `rms_gate` | 0.1–10 Hz | bounded relative RMS | 32,768 | **NEW 1000 updates** |
| `lowfreq_rms` | 0.1–4 Hz | bounded relative RMS | 32,768 | **NEW 1000 updates** |

The original `rope` checkpoint (zero extra time params) is a fifth reference.
**Neither original control is retrained, changed, or overwritten.**

At every action index j and noisy-action input embedding z_j:

```text
time feature = concat(sin(2π f_j τ), cos(2π f_j τ))
residual     = W · time feature,   W ∈ R^(1024×32), initially all zero
ordinary     = z_j + residual
RMS-calib    = z_j + clip(RMS(z_j) / mean_k RMS(z_k), 0.5, 2.0) · residual
```

All RMS terms use a detached FP32 view of z (no extra parameters/gradient
path). The new residual is inserted **before the entire Action Expert**.
Both training and denoising call the same embed_suffix() implementation.
Pretrained RoPE remains active in every attention layer.

All runs use the *same* 1000-update one-demo Uniform PDET plan, 2×A100,
batch 32/GPU, the same source `pi05_libero_base`, seed-1 imported
two-token VLM prompt, trainable full Action Expert, contiguous 50-step
action labels, and identical noise/time draw streams. No Event sampling
or experiment-II variable cadence is used.

## 1. Update the experiment branch

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/task7-time-input-spectral-1k
git switch -c feat/task7-time-input-spectral-1k FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

For an existing branch, use `git switch` then
`git merge --ff-only FETCH_HEAD` instead of `switch -c`.

## 2. Tests and GPU-free preflight

```bash
python -m pytest -q tests/scripts/test_pi05_task7_input_spectrum.py
DRY_RUN=true GPU_IDS=0,1 bash examples/training/train_pi05_task7_input_spectral.sh all
```

The script confirms the original 10-FPS task7 preparation, original
1000-step Uniform plan, source model/prompt, all three time residual
configuration flags, and two-GPU DDP recipe.

## 3. Train only THREE new 1000-update models

```bash
GPU_IDS=0,1 bash examples/training/train_pi05_task7_input_spectral.sh all
```

Uses BOTH GPUs for each model, **sequentially**, not three concurrent
training jobs. Checkpoints saved under:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/
  pi05-task7-timeinput-spectral-r1-<case>-task7-ep7-steps1000-aexpert-base-vlm2-P1-D0-E1-T1/
    checkpoints/001000/pretrained_model/
```

Only the new cases `lowfreq`, `rms_gate`, `lowfreq_rms` are trained.
To recover an unfinished case, pass its name instead of `all`.
Never overwrite or silently resume a partially created run; use an
explicit new `--repeat` only if the corresponding original controls
are also available with that repeat number.

Post-training **sample-exact auditing** checks each new case and matches
all 1000 rank-wise sample IDs, Flow noise and Flow time hashes to the
**previous** saved `rope` and `time_input` controls. The recorded
source checkpoint, 2-token prompt and PDET plan must also match. The
old and new runs have different Git source fingerprints by design; the
audit compares their actual sampling streams instead of pretending they
were trained from the same exact source file tree.

```bash
python -m lerobot.scripts.pi05_task7_input_spectrum audit all
```

## 4. Evaluate five models with GPU 0 and GPU 1

The **new evaluation seeds 6000,7000** are the default, because the
earlier 5000 result was already inspected and 2000 was used to choose
the 12cm shift. These are new perturbation/policy-random-number draws,
but still reuse the same official task7 initial-state indexes; they
are NOT wholly independent environments or a second demonstration.

```bash
DRY_RUN=true GPU_IDS=0,1 bash run_eval_libero_task7_input_spectral_2gpu.sh \
  --seeds 6000,7000 --shift-m 0.12 --shift-seed 20261103

GPU_IDS=0,1 bash run_eval_libero_task7_input_spectral_2gpu.sh \
  --seeds 6000,7000 --shift-m 0.12 --shift-seed 20261103
```

Two models run in parallel per phase, never more than two at once.
The launcher **reuses** old RoPE/Time-Input evals if the requested seed
already has valid raw results; for missing old seed results, it evaluates
the already trained control checkpoint and saves it under the original
experiment-I evaluation root. If a completed eval has a wrong physics
manifest, the script **fails** instead of combining mismatched evidence.

Each model runs 10 task7 episodes per seed, fixed deterministic object
shift, no image noise, and the original simulator control rate.

To compare against existing earlier seed cohorts:

```bash
GPU_IDS=0,1 bash run_eval_libero_task7_input_spectral_2gpu.sh \
  --seeds 1000,3000,4000,5000 --shift-m 0.12 --shift-seed 20261103
```

Every seed prints the five-case success-rate matrix in terminal. The
final report pools **success counts / total episodes** and computes
differences to the original Time-Input checkpoint, with no fabricated
significance claim.

Results for the three new cases:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-time-input-spectral/
  fixed-object-shift/.../r1-seed6000-n10-task7-only/
    summary-five.json
  pooled/summary-<config-hash>.json
  pooled/summary-<config-hash>.csv
```

Existing RoPE/Time-Input results stay in:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-timepath-qkv/
```

## 5. Reprint results WITHOUT GPUs

```bash
python -m lerobot.scripts.pi05_task7_input_spectrum report all \
  --seeds 6000,7000 --shift-m 0.12 --shift-seed 20261103
```

## Conclusions this experiment can and cannot support

- `lowfreq - time_input`: spectral band-limiting effect, same injection
  route and learned parameter budget.
- `rms_gate - time_input`: relative RMS gain effect, same original
  0.1–10Hz frequency basis.
- `lowfreq_rms - lowfreq - rms_gate + time_input`: frequency × RMS-gain
  interaction, all four time-on cells have exactly 32,768 time parameters.
- Compared with `rope`, any temporal arm also adds a learned position
  projection, so the difference does not prove the absence of time
  information in pretrained RoPE.

A 7/10 result is only one success ahead of 6/10. This diagnostic
cannot establish general VLA robustness without larger, independent
task and training-seed experiments. Don't select a new perturbation
magnitude after looking at the optimized results.
