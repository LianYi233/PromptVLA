# PDET repeats and fixed-task order controls

Run these queues in **two terminals** in `/home/wyn/VLAA/PromptVLA`, using the same
`lerobot` environment. Each experiment uses two GPUs, batch32 per GPU (global64),
bf16, token2, global seed0, 3000 updates, and the existing imported P0/P1 files.
Only the final **full checkpoint003000** is saved. Small probe/prompt snapshots and
scalar/input traces are deliberately retained for analysis.

```bash
# Terminal1: points1/2/3, GPU0/1. Four runs in this order.
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29541 REPEAT_ID=1 \
PDET_CASES="1011 0011 1001 1100" \
bash examples/training/train_pi05_pdet_followup.sh repeats \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true

# Terminal2: point4, GPU2/3. Two runs; separate rendezvous port.
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29542 REPEAT_ID=1 WITHIN_SEEDS="0 1" \
bash examples/training/train_pi05_pdet_followup.sh order \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

Checkpoints/logs use the existing `/data1/wyn/chkpt/2601-lerobot/{PromptVLA,logs}`
roots. CPU order plans default to `/data1/wyn/prompt-work/pdet-order-rep1/`.
`DRY_RUN=true` validates resources/outputs and builds the small CPU plans without
constructing a model. Set `OUTPUT_ROOT`, `LOG_ROOT`, `DATASET_ROOT`,
`BASE_PRETRAINED_PATH`, `TOKENIZER_PATH`, or `PROMPT_INIT_DIR` only when relocating.

| Queue | Run directory name |
| --- | --- |
| Repeat collapse | `pi05-pdet-rep1-base-vlm2-P1-D0-E1-T1` |
| Repeat P rescue | `pi05-pdet-rep1-base-vlm2-P0-D0-E1-T1` |
| Repeat E rescue | `pi05-pdet-rep1-base-vlm2-P1-D0-E0-T1` |
| Complete factorial slice | `pi05-pdet-rep1-base-vlm2-P1-D1-E0-T0` |
| Within-task order W0 | `pi05-taskfixed-rep1-W0-base-vlm2-P1-D0-E1-T1` |
| Within-task order W1 | `pi05-taskfixed-rep1-W1-base-vlm2-P1-D0-E1-T1` |

`rep1` is a repeat label, **not a random seed**. These repeats keep all original
P/D/E/T values; `1100` is the newly added combination. No output is overwritten or
silently skipped. After part of a queue succeeds, select only remaining
`PDET_CASES`/`WITHIN_SEEDS`. A directory from a failed run must be archived to a
new name before a from-scratch retry. A later full repeat can use `REPEAT_ID=2`.

## What the two queues measure

The repeat queue retains `pdet-step-rank-v1`: original sampler and flow draw math,
learning rate, warmup, batch size, initial prompt and optimizer settings. The
additional diagnostic forwards restore RNG and module states and use a temporary
prompt leaf outside DDP. Tests check non-interference; this is not a guarantee of
bitwise CUDA determinism. cuDNN settings are not changed relative to old runs.

The order queue is a **new** `pdet-occurrence-flow-v1` protocol. Its default:

1. D0 selects a finite192000-use pool from shuffled complete frame-index epochs.
   There is no extra padding at epoch boundaries. Thus this is not a claim of
   replaying the historical3000-step D0 trajectory, especially across epoch tails.
2. Each use gets a unique occurrence ID. A frame reused later gets a different ID.
3. Freeze the task ID in every global sample slot, including rank and batch slots.
4. W0 and W1 independently permute occurrences only among slots of the same task.
5. Gaussian epsilon and Beta time are generated on CPU from `(E/T seed, occurrence
   ID)` and transferred to the action device. They follow the occurrence regardless
   of step, rank or within-batch position. P1 is the same imported tensor.

Both arms have exactly the same instance multiset, exposure counts and task
schedule. The order intervention changes which episodes/frames within a task are
encountered together and when, relative to the fixed learning-rate schedule. It
does **not** separately isolate episode effects from frame effects or ordering
from within-task batch composition. D here labels the shared pool/task-plan seed;
W is the intervention. W0/W1 are randomized controls, **not easy/hard curricula**.
Difficulty-based scheduling is a subsequent experiment once this isolation test
is interpreted. Do not compare their scores to legacy1011 as a pure order effect:
their E/T generation protocol differs. Compare W0 against W1 within this queue.

## Diagnostics and audit

Every run records:

- `pdet/manifest-rank*.json`: factors, code revision, software versions, prompt
  hash and (for order runs) exact plan hashes.
- `pdet/inputs-rank*.jsonl`: all3000 batches on every rank, sample IDs, task IDs,
  episode/frame IDs, flow hashes, temporal quartiles and gripper changes.
  Order runs additionally hash noise/time for every occurrence.
- `training_diagnostics.jsonl`: ten-update windows of loss, gradient norm,
  parameter norm, actual optimizer update norm and relative update.
- `prompt_rank/`: exact fixed observations/noise/projections, projected Jacobian
  and NTK spectra, small prompt snapshots, and fixed-training-probe flow MSE.
  Measurements at0,50,100 and every250 updates through3000 use the original
  sixteen probe observations, four output projections and times0.2/0.5/0.8.
  These are **training-set diagnostics**, not a held-out validation estimate.
- `pdet_analysis/coverage.csv`: cumulative/window coverage at50,100,250,500,
  1000,2000,3000. Temporal quartiles are not labeled grasp/contact/release phases.
  A gripper change is an adjacent last-action-coordinate change greater than1e-6.
- `pdet_analysis/training_curve.csv` and `rank_and_fixed_probe_loss.csv`:
  analysis-ready curves. Rollout success must still be obtained with LIBERO eval.

Summaries run automatically after each successful training run. If the original
`pi05-pdet-...` run exists, `compare-inputs` also verifies its common recorded input
prefix (usually only8 steps) and saves `pdet_analysis/original_input_comparison.json`.
This check explicitly does not approve code/recipe equivalence across revisions.
The repeat queue
then runs the existing PDET audit; the order queue uses the new full-trace audit:

```bash
python -m lerobot.scripts.pdet_followup audit-order \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-taskfixed-rep1-W{0,1}-base-vlm2-P1-D0-E1-T1
```

This verifies every realized task slot and occurrence's frame/noise/time, not only
the first8 steps. The old PDET audit is intentionally unsuitable for order runs
because their stochastic protocol differs. Auditing new diagnostic repeats with
old commits may also reject recipe/code changes; do not bypass this and claim code
equivalence. Their original input sequences can be compared separately, while the
within-new-code repeats provide a consistent diagnostic experiment.

Resume via the standard trainer `--config_path=... --resume=true` requires the
original plan file, dataset, world size and batch size. The finite plan resumes at
the global occurrence offset; prefetched worker indices do not set resume state.
Since these recipes save only the final checkpoint, an interrupted pre-final run
normally has no resumable full checkpoint.

## Evaluate after training

Use the existing evaluator with the new prefixes (GPU IDs here are examples for
after training, not concurrent with the allocated training queues):

```bash
RUN_PREFIX=pi05-pdet-rep1 PDET_CASES="1011 0011 1001 1100" \
GPU_ID=0 CHECKPOINT_STEP=3000 EVAL_SEED=1000 EPISODES_PER_TASK=10 \
WAIT_FOR_CHECKPOINTS=false PROGRESS=true \
bash run_eval_libero_vlm_only_pdet_sweep.sh all

for w in 0 1; do
  RUN_PREFIX="pi05-taskfixed-rep1-W${w}" PDET_CASES="1011" \
  GPU_ID=2 CHECKPOINT_STEP=3000 EVAL_SEED=1000 EPISODES_PER_TASK=10 \
  WAIT_FOR_CHECKPOINTS=false PROGRESS=true \
  bash run_eval_libero_vlm_only_pdet_sweep.sh all || break
done
```
