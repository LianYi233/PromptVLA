# Order x noise-binding controls

Question: with a fixed training sample pool and task schedule, how much does the
effect of within-task sample order depend on its pairing with flow noise/time?

This is a controlled follow-up to the PDET and W0/W1 results. It is **not yet a
difficulty curriculum**. W0/W1 are random within-task permutations. Each changes
episode/frame order and within-task minibatch composition together. Task-level
ordering and easy/hard examples are not manipulated.

## Eight runs, two concurrent two-GPU queues

Every model: native base PI0.5, VLM-only token2, imported P1, fixed pool/task seed D0,
global seed0, bf16, compile=false, batch32 per GPU/global64, 3000 updates. Save only
the full checkpoint003000, while retaining small prompt/probe snapshots and logs.
Two jointly varied E/T banks (E1/T1 and E0/T0) each contain the same four cells:

| Queue | Order | Flow binding | Run prefix |
| --- | --- | --- | --- |
| GPU0,1 | W0 | occurrence | `pi05-orderbind-r1-W0-Bocc` |
| GPU0,1 | W1 | occurrence | `pi05-orderbind-r1-W1-Bocc` |
| GPU2,3 | W0 | slot | `pi05-orderbind-r1-W0-Bslot` |
| GPU2,3 | W1 | slot | `pi05-orderbind-r1-W1-Bslot` |

Append `-base-vlm2-P1-D0-E1-T1` or `-base-vlm2-P1-D0-E0-T0` to each prefix.
`r1` is only an experiment label; it does not change a seed. These are two noise
bank repeats, not eight independent random replicates or separate E/T effects.

## What is held constant

The shared finite pool contains 3000 x 32 x 2 = 192000 sample uses. An immutable
occurrence ID identifies each use; if a frame is reused, it has a new occurrence
ID for that use. W0/W1 only permute IDs within the SAME task's global slots.
Every global slot includes the step, distributed rank and within-batch position.

Both bindings use **exactly the same CPU generator, hash namespace, seed mapping,
noise shape and Beta time sampler**. The namespace remains the previously tested
`pdet-occurrence-flow-v1`, including for slot draws. A bank is indexed by the
integers 0..191999 and defines one pair `(epsilon_k, t_k)` for each key.

- `Bocc`: at slot j containing occurrence i, use bank key i.
- `Bslot`: at slot j containing occurrence i, use bank key j.

Each key is consumed once in every run. For a given E/T bank, all four runs use
the exact same multiset of sample uses AND exact same multiset of epsilon/time
pairs. At fixed W, changing B changes only their pairing. At fixed Bocc, changing
W preserves each sample-use/noise pairing; at fixed Bslot, changing W preserves
the noise/time sequence in global slots.

This slot arm is **not the historical step/rank-keyed GPU-noise protocol**. We do
not expect or require its score to reproduce the old1011 result of about25%.
The experiment isolates conditional order/pairing effects inside a common
generation protocol; it cannot by itself explain every difference from old runs.
Binding does not use sample content/difficulty and is not presumed superior.

Default plan directories are separate per queue to avoid races during concurrent
preparation, but the files have identical content/hashes for matching W labels.
Do not override PLAN_DIR to the same unprepared directory in concurrent queues.
The original P1 tensor is validated before any expensive model construction.

## Run

After previous jobs have finished, update the main checkout and activate the
existing `lerobot` environment. The scripts set PYTHONPATH to their own checkout;
no dependency or editable-install changes are necessary. Keep this checkout
unchanged while both queues run so recorded on-disk source hashes remain equal.

```bash
cd /home/wyn/VLAA/PromptVLA
git pull --ff-only origin main
```

Terminal1:

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29551 EXPERIMENT_ID=1 \
FLOW_BANKS="1 0" WITHIN_SEEDS="0 1" \
bash examples/training/train_pi05_order_binding.sh occ \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

Terminal2:

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29552 EXPERIMENT_ID=1 \
FLOW_BANKS="1 0" WITHIN_SEEDS="0 1" \
bash examples/training/train_pi05_order_binding.sh slot \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

Both queues sequentially run E1/T1-W0, E1/T1-W1, E0/T0-W0, E0/T0-W1. Outputs and
logs use `/data1/wyn/chkpt/2601-lerobot/{PromptVLA,logs}`. `DRY_RUN=true` validates
plans/resources/commands without loading the model. Existing output directories
cause a failure before any selected model is launched; they are never overwritten
or silently skipped. To continue only an unstarted last cell, for example, set
`FLOW_BANKS="0" WITHIN_SEEDS="1"` with the original EXPERIMENT_ID and queue.
A failed run with no full checkpoint needs a fresh output name or an explicitly
archived failed directory; do not turn on resume without a checkpoint.

## Diagnostics, full audit and evaluation

Same diagnostics as the preceding follow-up: all3000 input batches on both ranks,
prompt gradient/update norms at10-step log windows, rank/NTK and fixed TRAINING
probe flow loss at0,50,100,250,...,3000. After each run, `pdet_analysis/` is created.
The flow traces now also record `flow_key_ids`, allowing the actual binding to
be verified independently from the experiment name.

Each queue ends with a **partial** audit (`partial_inputs_passed`): it validates
its realized inputs but has only one binding. After both queues finish, run:

```bash
cd /home/wyn/VLAA/PromptVLA
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pdet_order_binding audit \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-orderbind-r1-W*-B*-base-vlm2-P1-D0-E*-T*
```

Expected: `audit=passed`, `complete_factorial=true`, `noise_banks=2`, eight runs.
This checks full step/rank coverage, the actual order, pool, task slots, per-key
draw hashes, code/recipe/initialization equality, and all factorial cells. It does
not prove bitwise numerical determinism. Do not mix old W0/W1 checkpoints into
this audit: all eight controls are rerun using the same new source snapshot.

Evaluate each queue with one GPU (two terminals; after training frees the GPUs):

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
GPU_ID=0 EXPERIMENT_ID=1 FLOW_BANKS="1 0" WITHIN_SEEDS="0 1" \
EVAL_SEED=1000 EPISODES_PER_TASK=10 PROGRESS=true \
bash run_eval_libero_order_binding.sh occ all

# In the second terminal use GPU_ID=2 and replace `occ all` with `slot all`.
```

All four suites,400 episodes/model, same evaluation seed and execution settings
as before. One in-place progress bar; final suite/overall rates on stdout, full
details in evaluation logs. A checkpoint's saved W/binding/PDET config must agree
with its directory label before evaluation. No trained model is loaded before
these checks, and the existing completed-checkpoint guard remains in effect.

## Interpretation

For each bank separately, compare W1-W0 under Bocc and under Bslot. Their
difference measures how the observed order effect depends on binding. Also
compare Bslot-Bocc at fixed W. Do not infer a universal ranking of P/D/E/T from
these conditional contrasts. Two permutations and two noise banks are a pilot,
not evidence for general stability or an optimal ordering policy.

If order differences persist under Bocc, investigate which examples appear early
and compare targeted curricula. If differences primarily emerge under Bslot,
investigate sample/noise pairing interactions. If all cells perform similarly,
the old collapse may depend on another part of the historical protocol or other
draw realizations; do not force a curriculum explanation. Rollout success is the
primary outcome; fixed training loss and rank/NTK curves are diagnostic evidence,
not substitutes for success or proof of the optimization mechanism.
