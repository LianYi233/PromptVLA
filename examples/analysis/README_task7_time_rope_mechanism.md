# Why a pretrained VLA's RoPE may benefit from an Action-Input Time Residual

**Scope:** Mechanism research for the existing PI0.5 1000-update LIBERO-long
task7 study. This code does NOT train any new models. It analyzes saved
per-episode outcomes and an existing trained Time-Input checkpoint, and
generates reproducible Matplotlib **PNG + SVG** figures backed by JSON data.

## Paper-motivation wording

Pretrained VLA action experts usually inherit general-purpose transformer
positional information via rotary position embeddings (RoPE). RoPE rotates
queries and keys according to token offsets, shaping the pairwise attention
scores. While effective for token-relative order, this attention-mediated
interface is not specifically optimized to expose *action-horizon-local*
temporal cues directly to every future continuous-action representation.
For a low-data continuation setting, making an action-local time signal
accessible at the expert input may provide a more readily adaptable
inductive bias. We thus **complement**, rather than remove, the pretrained
RoPE with a zero-initialized, learnable, low-rank temporal residual on
the noisy-action inputs. This intervention reaches all subsequent expert
layers. We distinguish empirical input-path utility from the unproven
claim that RoPE is inherently incapable of representing time.

At *fixed* FPS and 50-token chunk length, relative physical time
`j/fps` and action token index `j` encode the SAME underlying
information. A Time-Input improvement here is evidence for a better
**conditioning route/learned calibration**, not evidence that real
clock-time information was missing. The variable-cadence experiment
`feat/task7-physical-time-cadence-1k` is a distinct way to probe
non-equivalence of token-index position and true elapsed time.

## Relevant work and novelty limits

- **VideoRoPE: What Makes for Good Video Rotary Position Embedding?**
  ICML 2025, arXiv:2502.05173.
  https://arxiv.org/abs/2502.05173
  Investigates temporal-frequency allocation, temporal scaling, spatiotemporal
  indexing. Crucial context for the LOW-FREQUENCY time study, but not a
  direct VLA action-input residual.
- **T-RoPE: Time-Aware Rotary Position Embedding for Sequential
  Recommendation**, arXiv:2609.30576.
  https://arxiv.org/abs/2609.30576
  Illustrates token index vs real-time interval distinctions in a different
  domain. This does not prove fixed-FPS task7 has that distinction.
- **VLA-4D: Embedding 4D Awareness into Vision-Language-Action Models
  for SpatioTemporally Coherent Robotic Manipulation**, arXiv:2511.17199.
  https://arxiv.org/abs/2511.17199
  Already adds time to VLA perception AND spatiotemporal action
  representations. Do not claim first-ever temporal action encoding.
- **PaceVLA: Adaptive Temporal Action Quantization in
  Vision-Language-Action Models**, arXiv:2610.25961.
  https://arxiv.org/abs/2610.25961
  Varies temporal action decoding/execution granularity; a different method.
- **RoPE is Dead, Long Live RoPE: Towards Scalable Data-aware Positional
  Encodings**, arXiv:2609.34556.
  https://arxiv.org/abs/2609.34556
  Analyzes data-aware positional geometry and RoPE's own biases outside
  this VLA action-input setting.

## 1. Update repository

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git \
  feat/task7-time-rope-mechanism-visuals

git switch -c feat/task7-time-rope-mechanism-visuals FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

If already on the branch, use `git switch` followed by
`git merge --ff-only FETCH_HEAD` instead.

## 1b. Figure 0 — repeat ONE action 50 times (CPU-only mechanism illustration)

Before evaluating actual action predictions, isolate the temporal *input*
channel. Create a fixed **7-D normalized action vector** and copy it to all
**50 future positions**. No camera frames, task description, Gaussian Flow
noise, changed action targets, model training, or robot actions are involved.
Only the position in the chunk varies.

```bash
# Labelled illustration to check the plotting pipeline; NOT research evidence.
bash run_visualize_task7_constant_action.sh --illustrative \
  --output /tmp/task7-constant-action-illustration

# Automatically select the existing 1000-step time_input checkpoint from the
# saved task7 experimental plan (NOT the original RoPE-only checkpoint).
bash run_visualize_task7_constant_action.sh --from-task7-plan

# Alternatively, explicitly supply a trained Time-Input checkpoint directory:
bash run_visualize_task7_constant_action.sh \
  --checkpoint /path/to/time_input/checkpoints/001000/pretrained_model

python -m pytest -q tests/scripts/test_pi05_task7_constant_action.py
```

The trained mode only reads `config.json` and three tensors from
`model.safetensors`: `model.action_in_proj.weight`, its bias, and
`model.action_time_encoder.weight`. It reuses the **exact project's**
`RelativeActionTimeEncoding` module with the checkpoint's basis, learned
projection, frequencies, and gain mode. It does **not** load the 3B VLA,
require GPU, or run inference. The seven action values can be changed with
`--action 0.1,-0.1,0.05,0,0,0,0.5` (illustrative normalized values, not
physical robot commands); they are zero-padded to the policy's original
`max_action_dim` before the real input projection. For physical/global_dt checkpoints, `--delta-t`
sets the real action interval; the default is `1/fps`.
`--from-task7-plan` uses the same `--directory`, `--output-root`, and
`--repeat` settings as the established timepath launcher (defaults to repeat 1).

**Figure 0:** `fig00_constant_action_50.png/.svg`, with four panels:

1. The 50 rows of exactly identical actions (action dimensions 0–6).
2. PCA of 50 time-conditioned input embeddings: the RoPE-only / Zero
   **input** lies at one point; the learned Time-Input features can form a
   position-dependent trajectory. Color indicates the action index.
3. Pairwise cosine-similarity matrix of the *learned temporal residuals*.
4. L2 distance from the first action embedding for RoPE-only input,
   Time-Input Correct, and Time-Input Reversed (same weights, reverse order).

The file `constant_action_50_summary.json` records measured distances,
whether a signal was detected, checkpoint/weight fingerprints and caveats;
`constant_action_50_positions.csv` records 50 per-position measurements;
`constant_action_50_embeddings.npz` stores the actual action/input/temporal
vectors for further figures. If the learned temporal projection is zero or
collapses positions, the code reports **no detected signal**, rather than
pretending that the encoding works.

**Critical interpretation:** RoPE rotates Q/K later in the attention
mechanism, so an identical RoPE-only *expert input* does **not** mean RoPE
is incapable of distinguishing positions. This constant-action test can
prove that the additional **Time-Input path** introduces position-dependent
representations, but it cannot prove any advantage in action prediction.
The "RoPE-only / Zero" input is the **same Time-Input checkpoint with its
temporal residual bypassed**, not an independently trained RoPE checkpoint.
Use the held-out Correct/Zero/Reversed Flow-MSE probe in section 3 for that.

## 2. Run tests, then make ACTUAL rollout evidence — CPU only

```bash
python -m pytest -q tests/scripts/test_pi05_task7_time_evidence.py

# Uses existing task7 1000-step model evaluations. No GPU, no new rollouts.
bash run_analyze_task7_time_evidence.sh \
  --mode success --seeds 1000,3000,4000,5000
```

If only seed 5000 has been evaluated so far, use
`--seeds 5000`. The script **fails on missing or mismatched**
eval_info.json, checkpoint/physics manifest or seed, rather than
inventing results or silently fetching other runs.

**Figure 1**: `fig01_paired_task7_outcomes.png/.svg`.
A 4-row x (10×#eval_seeds)-column matrix of success/failure values
from **actual saved per-episode results** (not prior summary percentages).
Below: matched RoPE-vs-Time-Input win/loss counts. If the Time-Input
gain comes from just one sample, the figure makes that obvious.

Data: `paired_success.json`, including the exact 0/1 matrix,
paired counts and manifest hashes.

## 3. Probe the trained time pathway — GPU 0 (offline only)

```bash
GPU_ID=0 bash run_analyze_task7_time_evidence.sh \
  --mode probe --samples 8 --flow-time 0.5 --noise-seed 3000
```

This uses the 1000-step checkpoint for `rope` and `time_input`
from the ORIGINAL experiment-I branch naming scheme, not the experiment-II
variable-cadence model.

The script selects up to 8 anchor windows of **50 contiguous future
recorded actions**, chosen from *other* demos of the exact task7 language
and dataset task index, strictly excluding the selected training episode.
It refuses to claim a heldout test when no other demos exist.

For each identical anchor and fixed Flow Matching noise/timestep:

- `RoPE-only`: independently trained original RoPE checkpoint.
- `Time-Input correct`: trained 32->1024 time residual unchanged.
- `Time-Input zero`: SAME Time-Input weights, temporarily bypass the
  learned residual at inference only, RoPE unchanged.
- `Time-Input reversed`: SAME Time-Input weights, reverse the
  50 relative-time feature positions, RoPE unchanged.

The last two use **deliberate train/test feature mismatches**.
They diagnose *sensitivity to the learned time path*; they do not
isolate architecture generalization. `RoPE-only` is a **different
trained checkpoint**, which is a stricter but not same-weights contrast.

No robot control output is executed. The metric is normalized
Flow Matching velocity MSE against the same raw recorded 7-D targets
at each action index, with matched Gaussian noise and t=0.5.
Normalization mapping and processed target tensors are checked.

**Figure 2:** `fig02_learned_time_residual.png/.svg`
- Learned temporal input-residual L2 norm over 50 action positions.
- Cosine-similarity matrix of the learned position-residual vectors.
- These describe learned inputs, **not RoPE's full positional information**.

**Figure 3:** `fig03_time_input_counterfactual.png/.svg`
- Per-action velocity MSE: RoPE-only, correct/zero/reversed Time-Input.
- Per-anchor paired loss differences (including negative outcomes).
- Time residual to noisy-action embedding RMS ratio.
- Per-position change in predicted flow velocity under intervention.

All source anchor IDs, per-anchor MSE, per-step mean MSE, model file
fingerprints, noise seed, resulting figures and warnings are saved to
`offline_probe.json`.

Optionally both analyses:

```bash
GPU_ID=0 bash run_analyze_task7_time_evidence.sh \
  --mode all --seeds 1000,3000,4000,5000 --samples 8
```

## 4. Output

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-rope-time-input/
    fig00_constant_action_50.png
    fig00_constant_action_50.svg
    constant_action_50_summary.json
    constant_action_50_positions.csv
    constant_action_50_embeddings.npz
    fig01_paired_task7_outcomes.png
    fig01_paired_task7_outcomes.svg
    fig02_learned_time_residual.png
    fig02_learned_time_residual.svg
    fig03_time_input_counterfactual.png
    fig03_time_input_counterfactual.svg
    paired_success.json
    offline_probe.json
```

Use `--output /path/to/folder` to change this directory.

## 5. Interpretation for a rigorous introduction

Evidence supporting the *representation-calibration* hypothesis would
include multiple paired seeds with a Time-Input gain, nontrivial learned
time residuals, and deterioration of SAME-checkpoint held-out
velocity predictions when the time residual is zeroed or reversed,
with matched noise/timestamps. If the residual changes predictions
but not MSE, this only shows the model uses the extra signal; it is
not evidence for better forecasting.

If zeroing/reversing does not hurt, claims that Time-Input materially
supports action decoding are weak. If RoPE-only remains as good or
better on independent demos, Time-Input may be training-demo-specific
rather than generally more effective.

This is one task/one training demo on a model **already pretrained on
LIBERO**; different evaluation anchors within task7 do not establish
fully independent generalization. A paper-quality result also needs
multiple training seeds and tasks. Do not claim demonstrated RoPE
"failure", superiority over all RoPE variants, true-physical-time
handling at constant FPS, or statistically significant effects
without further controlled evidence.

The frequency aliasing of the separate 0.1–10 Hz *additive Fourier
feature bank* at 10 FPS is an issue of that supplemental module,
not evidence that pretrained Gemma RoPE itself aliases at 5 Hz.
