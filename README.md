# PromptVLA

**Prompt learning for frozen vision–language–action models.**

PromptVLA explores how a small set of learnable VLM prompt tokens can adapt a
frozen PI0.5 model for robotic manipulation. The current experiments study token
count, initialization sensitivity, independent sources of training randomness,
and the geometry of the prompt-to-action mapping on LIBERO.

For the paired GPU0/1 repeat/completion queue and GPU2/3 fixed-task-order controls,
see [PDET follow-up experiments](examples/analysis/README_pi05_pdet_followup.md).
Both keep two GPUs per experiment, batch32 per GPU, and save only the full3000-step checkpoint.

For the next same-bank **sample order x noise binding** factorial (eight runs,
two concurrent two-GPU queues), see [order/binding controls](examples/analysis/README_pi05_order_binding.md).

The default method trains **VLM prompts only**. The vision encoder, VLM backbone,
action expert, action projections, and time-conditioning modules remain frozen.
There are no action prompts, CABO, action priming, or conditioning bridge in this
experimental recipe. For PI0.5, N prompt tokens contain N × 2048 trainable parameters.
This reduces trainable parameters and per-task adaptation storage; inference still
requires the full backbone.

FrameSkip 风格课程实验：见 [评分、双卡训练与评估说明](examples/analysis/README_frameskip_curriculum.md)。固定 P1/D0、VLM token=2，比较 U/R/S/C 四种安排与两个 ε/t bank；S/C 严格共享样本使用实例和噪声。

关键动作加权课程：见 [W0-Bocc 基线、关键窗口与随机对照](examples/analysis/README_pi05_keyframe_curriculum.md)。复用旧 W0 的全部样本顺序和 ε/t 配对，只改变动作时间步的损失权重；包含 B/K2/R2 和前期／后期加权课程，支持四卡上两条独立双卡队列。

预训练来源对照：见 [重复 W0-Bocc，并比较 pi05_libero_base 与 pi05_base](examples/analysis/README_pi05_source_compare.md)。固定同一份 P1 prompt、D0/W0 样本及 E1/T1 噪声，只替换原始权重；两组共享 LIBERO 配置与处理器，四卡并行训练两个模型。

从难到易的顺序实验：见 [固定 LIBERO 起点与原样本/噪声配对，仅按旧评分降序训练](examples/analysis/README_pi05_hard_to_easy.md)。以本轮 source-libero 为对照，复用评分缓存，在各 task 内调整原 W0 样本使用顺序；单组双卡、VLM2、P1/D0/E1/T1、3000步，无重采样或loss加权。
设置 `TRAIN_ACTION_EXPERT=true` 可在同一入口训练 VLM prompt 和完整动作生成网络（动作 Transformer、动作投影、时间 MLP）；视觉/VLM主干仍冻结，模型名增加 `aexpert`，评估时使用相同开关。默认false保留原prompt-only实验。

可调曝光比例的双向课程：见 [易→难多学简单样本 / 难→易多学困难样本](examples/analysis/README_pi05_ratio_curriculum.md)。新入口 `train_pi05_ratio_curriculum.sh` 默认训练VLM2+完整action expert，用 `CURRICULUM_DIRECTION` 控制顺序、`EASY_PERCENT` 控制简单样本使用占比（默认70%/30%），双卡×batch32，`FLOW_STEPS=6000`默认只存最终006000。四卡可并行两组；整个预算只进行一次课程，不在中途重启。重采样会改变样本与噪声的配对，记录为独立干预；设置3000步仍兼容旧计划/名称。

97.75%参考设置的四轮消融：见 [一次启动8组顺序、比例和训练模块对照](examples/analysis/README_pi05_curriculum_ablation.md)。`train_pi05_curriculum_ablation.sh` 自动在GPU0/1和GPU2/3并行每轮两组、逐轮接续，默认r2、每组6000步仅保存最终模型。新增task内随机顺序、保留并冻结P1的expert-only及prompt-only对照，同一比例保留样本使用池与Bocc噪声配对，全部结束自动跨组审计。

D/E/F 新种子重复：见 [D1/E1/F1、D2/E2/F2 一次启动说明](examples/analysis/README_curriculum_seed_repeat.md)。保持50/50、VLM2+完整Action Expert、6000步和每组双卡×batch32，默认使用两套新P/D/E/T/global/S种子2、3；组内匹配样本池和Bocc噪声，只改变课程顺序。自动导出初始prompt、四卡分三轮训练六个模型，每次只保存最终006000。

两样本顺序对照：见 [同一对简单/困难观测，交换两个3000步阶段](examples/analysis/README_pi05_two_frame.md)。复用评分缓存，选同 task/episode 的两个完整、不重叠动作窗口；E2H/H2E 各训练6000步、曝光各半，固定帧与噪声配对，仅训练VLM2+完整action expert。四卡并行两组，记录两个训练样本各自的固定损失，只保存最终006000。

LIBERO-PRO / PLUS 泛化评估：见 [完整动作网络checkpoint的独立基准入口](examples/analysis/README_libero_robustness.md)。`run_eval_libero_robustness.sh` 校验并选择指定fork，加载保存的VLM2+action expert及处理器，支持任务级恢复、单行进度、按扰动类别汇总；PLUS严格检查10030个变体，PRO可明确选择已备齐资源的维度，不把部分评测标成全量。

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
| LIBERO assets | `LIBERO_ASSETS_PATH=/data1/datasets/libero-assets` (setup script) |
| Exported prompt initializations | `PROMPT_INIT_DIR=/data1/wyn/prompt-work` |
| Checkpoints (training / evaluation) | `OUTPUT_ROOT` / `CKPT_ROOT`: `/data1/wyn/chkpt/2601-lerobot/PromptVLA` |
| Training logs | `LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs` |
| Evaluation logs | `EVAL_LOG_ROOT`: `$LOG_ROOT/eval` |
| Evaluation results | `BASE_OUTPUT`: `$CKPT_ROOT/eval` |
| Temporary files | `TMPDIR`: `$OUTPUT_ROOT/.runtime/tmp` |
| Compiler caches | `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`: `$OUTPUT_ROOT/.runtime/cache/{torchinductor,triton}/$RUN_NAME` |

Prepare assets and fresh local token=2 initializations in the configured environment:

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
bash examples/setup/prepare_promptvla_local.sh assets
GPU_IDS=0,1 BATCH_SIZE=32 bash examples/setup/prepare_promptvla_local.sh inits
```

The asset step downloads the complete [LeRobot LIBERO assets](https://huggingface.co/datasets/lerobot/libero-assets)
with one worker, validates key resources, links the active installation's `assets`,
and updates `LIBERO_CONFIG_PATH/config.yaml` (default `/home/wyn/.libero`). Existing
asset locations/configs are backed up. Re-run after a download interruption; use
`python -m lerobot.scripts.setup_libero_assets --skip-download` to configure an
already downloaded directory offline. This is a resource check, not a simulator test.

The initialization step follows normal `pi05_libero_base` construction for seeds 0 and 1
and exports `pi05_libero_base-vlm2-init-seed{0,1}.safetensors`. It loads the full model but
performs zero updates, creates no optimizer/probes/full checkpoint, and disables
W&B. The files contain prompt hashes and construction provenance. These are **new
local initializations**, not verified reconstructions of the historical server's
prompts. Use the same pair throughout the new P/D/E/T experiment.
Existing files are never overwritten; after a partial export select the missing
seed with `INIT_SEEDS=1 ... inits`. Ordinary VLM-only training does not require these files.

Existing `pi05-base-vlm2-init-seed{0,1}.safetensors` files refer to the LIBERO source,
not the generic `pi05_base`. Rename them without changing tensor values or provenance:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.export_pi05_prompt_init rename-libero-inits /data1/wyn/prompt-work
```

The CPU-only migration validates the recorded source and keeps old names as relative
symlinks for historical configs. Launchers prefer explicit names and still accept
unmigrated legacy files; conflicting copies are rejected. The weights-only source
comparison deliberately shares the SAME saved P1 tensor across both backbones.
Do not regenerate P1 separately for the generic backbone in that comparison.

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
| Download/configure assets and export local seed0/seed1 prompts | `examples/setup/prepare_promptvla_local.sh` |
| Single VLM-only run | `examples/training/train_pi05_vlm_only.sh` |
| Token sweep, base / finetuned | `examples/training/train_pi05_vlm_only_token_sweep.sh` |
| Effective-rank / NTK monitoring | `examples/training/train_pi05_vlm_only_rank_sweep.sh` |
| Independent P/D/E/T experiment | `examples/training/train_pi05_vlm_only_pdet_sweep.sh` |
| Evaluate one checkpoint | `run_eval_libero_vlm_only.sh` |
| Evaluate a token sweep | `run_eval_libero_vlm_only_token_sweep.sh` |
| Evaluate P/D/E/T runs in order, waiting for final checkpoints | `run_eval_libero_vlm_only_pdet_sweep.sh` |

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
