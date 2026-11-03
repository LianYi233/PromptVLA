# Task7 · 1000-step Temporal Path Ablation: RoPE / QK / V / Input

## Research question and interpretation

This is **experiment I** of the proposed VLA time-conditioning study:
whether injecting action-chunk-relative time into action **content features**
is more effective than modifying first-layer **Q/K attention routing**.
It **does not** test whether physical time carries more information than a
fixed-rate position index (that requires experiment II's variable-Δt data).

The current π0.5 already uses pretrained Gemma RoPE in its Attention
projections, including action tokens. **All four conditions preserve RoPE**.
Nothing replaces RoPE or changes the attention mask, action targets, or
vision-language prefix.

| Case | Additional temporal information | Extra parameters |
|---|---|---:|
| `rope` | none; original RoPE only | 0 |
| `time_qk` | first Action Expert layer Q and K projections | 32,768 |
| `time_v` | first Action Expert layer V projection only | 32,768 |
| `time_input` | action input embeddings before the expert | 32,768 |

The latter three use the **same** 32-D sine/cosine basis
`φ(j/fps)`, for `j=0,...,49`, and the same zero-initialized learned
matrix `W ∈ R^(1024×32)`. RoPE itself acts on Q/K in every layer.
The input route adds `Wφ` to the original action input embeddings.
The Q/K and V routes transform `Wφ` with the *already existing,
learned* first-layer q_proj/k_proj or v_proj weight, and add that result
only to the selected native projections. **No new per-layer trainable
weights** and no changes to HF checkpoint parameter key names.

Time-QK and Time-V deliberately intervene only in the **FIRST**
Action Expert layer. They retain the trained transformer's later
propagation, but are not the same as adding time independently to all
layers. Consequently the experiment compares practical intervention
sites, not a theoretically identical amount of signal across routes.
Stateless forward hooks are active in the joint training path **and**
the cached-prefix suffix-only inference path, including gradient
checkpoint replay. A shape guard fails if the expert receives unexpected
token lengths. The experiment requires zero Action Expert prompt tokens.

The source is already trained on LIBERO: report the study as *single
demonstration continued adaptation*, **not** genuine one-shot learning
from an untrained base.

## Source data and compute controls

This study reuses the **original 1000-step Uniform plan** prepared in:

```
/data1/wyn/prompt-work/gripper-event-task7-1200/base/uniform.npz
```

It does **not** use the later additive 200 Event steps or the
`.../additive` plan. The prepared plan selects one complete LIBERO-long
task7 demonstration (previous selection: episode 7) and exact, fixed
contiguous 50-action chunks. All four cases consume the same 64,000
sample occurrences (1000 updates × 2 GPUs × per-GPU batch 32).
They also share the original seed-1 initialized **2-token VLM prompt**,
the same `pi05_libero_base` source checkpoint, sample stream, training
seed 0, Flow noise, Flow timestep draws, and 1000-step schedule.
The VLM backbone stays frozen while the 2 VLM prompts and complete
Action Expert are optimized as in the previous experiment.

Each run saves `checkpoints/001000/pretrained_model`, with a new prefix
`pi05-task7-timepath-r1-<case>-task7-epN-steps1000-...`.
The 1200-step additive results/checkpoints and previous clean/shifted
eval folders are **not overwritten**.

## 1. Fetch and activate the experiment branch

```bash
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git \
  feat/task7-timepath-qkv-1k
git switch -c feat/task7-timepath-qkv-1k FETCH_HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

If already on the branch, update with `git merge --ff-only FETCH_HEAD`
after fetching.

## 2. Static checks and preflight

```bash
python -m pytest -q tests/scripts/test_pi05_task7_timepath.py
python -m lerobot.scripts.pi05_gripper_event check \
  /data1/wyn/prompt-work/gripper-event-task7-1200/base

DRY_RUN=true GPU_IDS=0,1 \
  bash examples/training/train_pi05_task7_timepath.sh all
```

The dry run performs model/dataset/provenance preflight without launching
the GPU optimizer. Verify the source path, plan, token source, checkpoint
names, `FLOW_STEPS=1000`, data/noise seeds, and time injection flag.

## 3. Train four cases, two DDP GPUs per model (SEQUENTIAL)

```bash
GPU_IDS=0,1 bash examples/training/train_pi05_task7_timepath.sh all
```

This runs all four conditions **one after another**, each occupying GPU
0 and GPU 1 as a 2-GPU DDP job. This is not four jobs running in parallel.
For individual recovery (run must not already exist), choose:
`rope`, `time_qk`, `time_v`, or `time_input` instead of `all`.

When four checkpoints are complete:

```bash
python -m lerobot.scripts.pi05_task7_timepath_experiment audit all
```

The audit verifies the enabled mode in the saved checkpoint, existence
and 1024×32 shape of each time tensor, checkpoint training budgets and
the per-step actual sampled indices, Flow noise and Flow timestep records
among all four models. Original RoPE and QKV weights remain in their
original architecture and checkpoint key names.

## 4. Fixed-displacement evaluation, two GPUs, two phases

Run the SAME 12cm task7 displacement as before. Each model is evaluated
on the *same ten init-state indexes, object displacement keys and policy
random streams* for a given seed. The pilot seed 2000 was used for
selecting displacement; report it separately, not as independent test
evidence. Prefer independent test seeds 1000, 3000, 4000, 5000.

```bash
# Inspect all four inference commands, without GPU evaluation:
DRY_RUN=true GPU_IDS=0,1 \
  bash run_eval_libero_task7_timepath_2gpu.sh \
    --repeat 1 --eval-seed 3000 --episodes 10 \
    --shift-m 0.12 --shift-seed 20261103

# Execute two concurrent models at a time:
GPU_IDS=0,1 \
  bash run_eval_libero_task7_timepath_2gpu.sh \
    --repeat 1 --eval-seed 3000 --episodes 10 \
    --shift-m 0.12 --shift-seed 20261103
```

**Phase 1:** GPU0 `rope` and GPU1 `time_qk` run concurrently.

**Phase 2:** GPU0 `time_v` and GPU1 `time_input` run concurrently.
The launcher blocks until phase 1 completes; if any case fails, no
combined report is generated. Two GPUs must each have sufficient free
memory; default preflight requires 20,000 MiB. Do not run while those
GPUs are still training.

After the first seed works, repeat four independent eval seeds:

```bash
for seed in 1000 3000 4000 5000; do
  GPU_IDS=0,1 bash run_eval_libero_task7_timepath_2gpu.sh \
    --repeat 1 --eval-seed "$seed" --episodes 10 \
    --shift-m 0.12 --shift-seed 20261103
done
```

Each model runs **only task7, ten episodes, no observation/image noise**.
Every evaluation directory includes shift magnitude, shift seed and
eval seed, with a saved JSON physics manifest. Results are resumable and
cannot silently mix different model checkpoints or perturbation code.
They are separate from earlier 1200-step model and benchmark folders:

```
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/task7-timepath-qkv/
```

Log files:

```
/data1/wyn/chkpt/2601-lerobot/logs/task7-timepath-qkv/
```

Each run produces `summary-all.csv` and `summary-all.json` containing
per-case successes plus `qk_minus_rope_pp`,
`v_minus_rope_pp`, `input_minus_rope_pp`, and
`input_minus_qk_pp`.

## Interpretation and next experiment

If Time-Input consistently wins over Time-QK with matched samples,
it supports direct action-embedding temporal conditioning as an effective
*inductive bias*. If Time-V also improves over QK, it suggests a value
path mechanism. These outcomes do **not** prove a fundamental limitation
of RoPE: at fixed FPS, `j/fps` is a deterministic transform of
token position, already available to a sufficiently expressive model.

This first experiment is restricted to **one training trajectory and
one task**, hence is a mechanistic diagnostic, not a general-purpose
benchmark. For a paper, repeat training seeds and expand to other long
tasks. Only if this first experiment is informative should we implement
experiment II, using multiple actual action time intervals and matched
offline metrics to test whether physical time adds information beyond
a token index.
