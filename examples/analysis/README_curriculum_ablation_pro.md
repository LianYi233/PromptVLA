# A-H curriculum ablation: four-GPU LIBERO-PRO evaluation

Use the existing r2, 6000-step models. This launcher does not train anything.
All models use the same fork, resources, seed, task coverage, action execution length,
saved checkpoint processors, image resolution and simulator horizons.

| GPU | First | Then |
|---|---|---|
| 0 | A: H2E 30/70 joint | E: E2H 50/50 joint |
| 1 | B: E2H 30/70 joint | F: RND 50/50 joint |
| 2 | C: RND 30/70 joint | G: H2E 30/70 expert_only |
| 3 | D: H2E 50/50 joint | H: H2E 30/70 prompt_only |

`G` retains its frozen P1 prompt. `H` retains its frozen action expert. Each loads
its complete trained checkpoint, never a replacement base action network.
Scope-aware validation lives in a separate helper so the existing benchmark
protocol helper/hash remains unchanged for other PRO/PLUS resume directories.

## Launch once

Activate the environment that already runs LIBERO-PRO (e.g. `promptvla-eval`).

```bash
conda activate promptvla-eval
cd /home/wyn/VLAA/PromptVLA

export CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA
export BASE_OUTPUT="$CKPT_ROOT/eval/curriculum-ablation-r2"
export EVAL_LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs/eval/curriculum-ablation-r2
export BENCHMARK_ROOT=/home/wyn/VLAA/LIBERO-PRO
export TOKENIZER_PATH=/home/wyn/models/paligemma-3b-pt-224
export GPU_IDS=0,1,2,3 CELLS="A B C D E F G H"
export REPEAT_ID=2 CHECKPOINT_STEP=6000 EVAL_SEED=1000 EPISODES_PER_TASK=10
export PRO_DIMENSIONS=object,swap,lan,task PROGRESS=true DRY_RUN=false
mkdir -p "$EVAL_LOG_ROOT"

nohup bash run_eval_libero_curriculum_ablation_pro.sh run \
  >> "$EVAL_LOG_ROOT/pro-launch.log" 2>&1 < /dev/null &
echo "PRO queue PID=$!"
```

Every selected checkpoint is preflighted before any GPU evaluation starts. The
preflight creates the shared immutable benchmark manifest sequentially, avoiding
concurrent manifest creation. A per-batch lock rejects duplicate queue launches.
Each GPU runs one model at a time; failure in one cell does not discard other cells.
After a failed job, the next queued model can still run. Aggregate exit status is
nonzero if any selected job failed. A failure is never reported as a zero success rate.

The default scope is **four PRO dimensions**: object, swap, lan and task, with
4 suites × 4 dimensions × 10 tasks × 10 episodes = **1600 episodes per model**,
12800 for all eight. This is not five-dimensional full PRO. `env` is not tested.
Only select `PRO_DIMENSIONS=all` if the complete verified env resources are present;
that changes coverage to 2000 episodes/model and uses a distinct result directory.

## In-place progress and summary

In the same environment/settings:

```bash
watch -n 10 'bash run_eval_libero_curriculum_ablation_pro.sh status'
```

Ctrl-C stops only the watch command. The background evaluation continues.
The queue checks all models first, so early status may say `pending/preflight`.

```bash
bash run_eval_libero_curriculum_ablation_pro.sh summary
```

This reads actual per-task results, checks model identity, evaluation settings,
protocol identity, complete task coverage and boolean episode outcomes. It prints
both four-base-suite and perturbation-category tables. Suite scores aggregate the
selected perturbations for that base suite. Missing/partial results remain blank.
The run command also produces the summary automatically after all queues finish.

Fixed default batch directory (no dependence on the old `$EVAL_BATCH_DIR`):

```text
/data1/wyn/chkpt/2601-lerobot/logs/eval/curriculum-ablation-r2/
  pro-batch-seed1000-ep10-dimsobject-swap-lan-task/
    A.preflight.log ... H.preflight.log
    A.result.txt ... H.result.txt
    A.progress.log ... H.progress.log
    status.json
    summary.txt
    summary.json
```

Full task-level data reside under:

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/curriculum-ablation-r2/
  pro/seed1000-ep10-dimsobject-swap-lan-task/
```

To resume, rerun the same launch after the previous queue has exited; completed
task records are reused by the existing evaluator. To rerun only certain cells,
set e.g. `CELLS="G H" GPU_IDS=0,1`; restore `CELLS="A B C D E F G H"` for the
full summary. Keep all other protocol/model settings identical. Do not change
evaluation code, assets or environment while the queue is running.

## Interpretation

Primary comparison: A minus C, same 30/70 exposure, trainable modules and training
budget. D minus F tests the corresponding contrast at 50/50. B/C and E/F give
easy-to-hard contrasts. A/G/H compare trainable scopes, not order alone.
Analyze paired task/episode outcomes under this common evaluation protocol.
More test episodes do not substitute for independent training seeds; these runs
alone cannot establish a training-seed-robust curriculum advantage.
