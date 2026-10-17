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

## Existing AutoDL environment

```bash
cd /root
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LianYi233/PromptVLA.git
cd PromptVLA
conda activate lerobot
python -m pip install --no-deps -e .
python -c 'import lerobot; print(lerobot.__file__)'
```

The printed package path should be under `/root/PromptVLA/src/lerobot/`.
`--no-deps` reuses the working environment and avoids changing the experimental
software stack. Complete any running experiments in the old checkout before
switching the environment's editable install. A fresh environment needs the
dependencies described in the [upstream setup guide](README_LEROBOT.md) and the
LIBERO evaluation dependencies.

The clone command skips inherited Git LFS test fixtures. Their pointers are
preserved in this snapshot; the LFS object store was not copied. These fixtures
are not used by the PI0.5/LIBERO experiment scripts, which load the external
models, datasets, and assets listed below.

| Resource | Current location |
| --- | --- |
| Base model | `/root/autodl-fs/wyn/models/pi05_libero_base` |
| Finetuned model | `/root/autodl-fs/wyn/models/pi05_libero_finetuned_v044` |
| Tokenizer | `/root/autodl-fs/wyn/models/google/paligemma-3b-pt-224` |
| LIBERO dataset | `/root/autodl-fs/wyn/datasets/libero` |
| LIBERO assets | `/root/autodl-fs/wyn/libero-assets` |
| Exported prompt initializations | `/root/autodl-fs/wyn/prompt-inits` |
| Checkpoints | `/root/autodl-tmp/chkpt/2601-lerobot/vlm-only` |
| Training logs | `/root/autodl-tmp/logs/vlm-only` |
| Evaluation results | `/root/autodl-tmp/eval/vlm-only` |

The migration does not move model weights, datasets, or experiment outputs.
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
