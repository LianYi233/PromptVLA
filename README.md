# PromptVLA

**Prompt learning for frozen vision–language–action models.**

PromptVLA explores how a small set of learnable VLM prompt tokens can adapt a
frozen PI0.5 model for robotic manipulation. The current experiments study token
count, initialization sensitivity, independent sources of training randomness,
and the geometry of the prompt-to-action mapping on LIBERO.

The default method trains **VLM prompts only**. The vision encoder, VLM backbone,
action expert, action projections, and time-conditioning modules remain frozen.
There are no action prompts, CABO, action priming, or conditioning bridge in this
experimental recipe. For PI0.5, N prompt tokens contain N × 2048 trainable parameters.
This reduces trainable parameters and per-task adaptation storage; inference still
requires the full backbone.

## Project and implementation

This independent repository continues the `vlm-only` work from
`LianYi233/lerobot060`, including the P/D/E/T experiment at commit
`ac06d51011450f76724c64ed3b7ef1e3d93aa523`. This repository starts from that code
snapshot; earlier Git history remains in the source repository. Upstream license
notices are retained. The default development branch is `main`.

The implementation is based on Hugging Face LeRobot. The Python package and CLI
names remain `lerobot`, `lerobot-train`, and `lerobot-eval` so existing checkpoints,
processor configurations, and experiment scripts remain compatible. Install this
checkout into the active environment before running it.

## Local server with four A100 GPUs

Current experiments run on a local server with four A100 GPUs. The checkout lives
at `/home/wyn/VLAA/PromptVLA`:

```bash
mkdir -p /home/wyn/VLAA
cd /home/wyn/VLAA
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LianYi233/PromptVLA.git
cd /home/wyn/VLAA/PromptVLA
```

If the server already has a configured `lerobot` environment, install this checkout
into it:

```bash
conda activate lerobot
python -m pip install --no-deps -e .
python -c 'import lerobot; print(lerobot.__file__)'
```

The printed package path should be under `/home/wyn/VLAA/PromptVLA/src/lerobot/`.
`--no-deps` reuses the working environment and avoids changing the experimental
software stack. Complete any running experiments in the old checkout before
switching the environment's editable install. A fresh environment needs the
dependencies described in the [upstream setup guide](README_LEROBOT.md) and the
LIBERO evaluation dependencies.

The clone command skips inherited Git LFS test fixtures. Their pointers are
preserved in this snapshot; the LFS object store was not copied. These fixtures
are not used by the PI0.5/LIBERO experiment scripts, which load the external
models, datasets, and assets.

### Local resource paths

The VLM-only launchers default to the following paths. Explicit environment
variables still override them; unset stale AutoDL overrides in an existing shell.

| Resource | Default location / setting |
| --- | --- |
| Base model | `BASE_PRETRAINED_PATH=/home/wyn/models/lerobot/pi05_libero_base` |
| Finetuned model | `FINETUNED_PRETRAINED_PATH=/home/wyn/models/lerobot/pi05_libero_finetuned_v044` |
| Tokenizer | `TOKENIZER_PATH=/home/wyn/models/paligemma-3b-pt-224` |
| LIBERO dataset | `DATASET_ROOT=/home/wyn/datasets/libero` |
| Checkpoints (training / evaluation) | `OUTPUT_ROOT` / `CKPT_ROOT`: `/data1/wyn/chkpt/2601-lerobot/PromptVLA` |
| Training logs | `LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs` |
| Evaluation logs | `EVAL_LOG_ROOT`: `$LOG_ROOT/eval` |
| Evaluation results | `BASE_OUTPUT`: `$CKPT_ROOT/eval` |
| Temporary files | `TMPDIR`: `$OUTPUT_ROOT/.runtime/tmp` |
| Compiler caches | `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`: `$OUTPUT_ROOT/.runtime/cache/{torchinductor,triton}/$RUN_NAME` |

LIBERO assets and exported prompt initialization locations are still to be supplied.
Configure LIBERO's local asset paths (including any package asset symlinks) before
evaluation. `LIBERO_CONFIG_PATH` defaults to `/home/wyn/.libero`; that directory alone
does not configure the assets. P/D/E/T training requires an explicit `PROMPT_INIT_DIR`
containing the exported seed0/seed1 files and stops before training if it is unset.
Ordinary VLM-only training does not require those files.

The availability of four GPUs does not change the experiment recipe automatically;
the token/rank/PDET sweeps still default to two GPUs and batch 32 per GPU. Keep
world size and batch size fixed across controlled comparisons.

The migration does not move model weights, datasets, or experiment outputs.
When evaluating old AutoDL checkpoints, their source-path provenance check may need
a verified migration: `train_config.json` records the original absolute model path.
W&B continues to use the `prompt-learning` project for comparison with existing runs.

## Experiment entry points

| Experiment | Entry point |
| --- | --- |
| Single VLM-only run | `examples/training/train_pi05_vlm_only.sh` |
| Token sweep, base / finetuned | `examples/training/train_pi05_vlm_only_token_sweep.sh` |
| Effective-rank / NTK monitoring | `examples/training/train_pi05_vlm_only_rank_sweep.sh` |
| Independent P/D/E/T experiment | `examples/training/train_pi05_vlm_only_pdet_sweep.sh` |
| Evaluate one checkpoint | `run_eval_libero_vlm_only.sh` |
| Evaluate a token sweep | `run_eval_libero_vlm_only_token_sweep.sh` |

The P/D/E/T experiment uses token=2 and independently controls the imported prompt
initialization (P), sample order (D), Gaussian flow noise (E), and Beta flow time (T).
The seven default runs use two GPUs per model, 3000 updates, and save only the final
full checkpoint. Small prompt snapshots, rank measurements, and random-input audit
records are retained. This is an independent-stream protocol, not a replay of a
historical global-seed trajectory.

See the complete commands and interpretation in:

- [P/D/E/T experiments](examples/analysis/README_pi05_pdet.md)
- [Effective-rank and NTK analysis](examples/analysis/README_vlm_prompt_rank.md)
- [VLM-only training and evaluation](examples/training/vlm_only_libero.md)

## Provenance and license

PromptVLA is derived from [LeRobot](https://github.com/huggingface/lerobot).
The inherited implementation and notices are retained under [Apache-2.0](LICENSE).
The previous repository README is preserved as [README_LEROBOT.md](README_LEROBOT.md).
Historical support for other models and training recipes remains in the codebase;
the experiment entry points above define the PromptVLA VLM-only setup.
