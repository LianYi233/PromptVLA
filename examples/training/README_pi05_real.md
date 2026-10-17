# May-pick-and-place 真机数据训练（prompt-ablation）

**在 `piper` 分支的 AutoDL 四任务重训入口：**
`bash examples/training/train_piper_autodl.sh all`，见
[AutoDL 重训说明](../piper/README_TRAIN_AUTODL.md)。该入口默认正式 flow 12000 步、单卡、
仅保存 6000/9000/12000 步、部署执行段 8 步；下文原入口的默认值仍为 flow 3000 步。

入口：`bash examples/training/train_pi05_real.sh TASK [VARIANT] [SEED] [额外训练参数...]`。
默认 `VARIANT=full_reference`、`SEED=0`，复用本分支的 PI05 prompt-only 训练流程。
VLM、action expert 和投影层保持冻结；只训练所选 prompt。数据格式是 **LeRobot v3.0**，
无需转成 LIBERO 格式，也不需要接入机器人即可进行离线训练。

## 数据目录与任务

`DATASET_BASE` 指向 `May-pick-and-place`，每个任务目录内都有 `data/`、`meta/`、`videos/`。

| TASK | 子目录 |
| --- | --- |
| `1` | `1-put_the_apple_on_the_yellow_plate` |
| `2` | `2-remove_the_cuboid_from_blue_plate` |
| `3` | `3-move_the_tennis_from_yellow_plate_to_blue_plate` |
| `4` | `4-pick_the_red_cube_into_the_yellow_plate` |
| `all` | 按 1、2、3、4 的顺序分别训练，得到四个独立模型 |

也可以用完整子目录名代替编号。`all` **不是四任务混合训练**，不会串联前一个任务的权重。
当前 dataset factory 尚未启用多数据集联合训练。

每个子目录应包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、
`meta/episodes/chunk-*/file-*.parquet`、`data/chunk-*/file-*.parquet` 及视频文件。
训练时的语言指令来自 `meta/tasks.parquet` 及帧里的 `task_index`，不会从文件夹名覆盖指令。
相机字段来自 `meta/info.json` 的 `observation.images.*`，state/action 维度由数据推导，
支持不超过 32 维的一维向量；内部沿用预训练模型的 32 维 padding。
保留数据中的动作单位和表示方式，不自动进行度/弧度、关节/末端或绝对/相对动作转换。

## 环境和路径

在安装了本分支的 LeRobot 环境中执行（Python 3.12+）。首次安装可使用：

```bash
pip install -e '.[training,pi]'
```

在仓库目录设置本机的实际路径，以下是 AutoDL 示例：

```bash
export DATASET_BASE=/root/datasets/May-pick-and-place
export PRETRAINED_PATH=/root/autodl-tmp/models/pi05_libero_base
export TOKENIZER_PATH=/root/autodl-tmp/models/google/paligemma-3b-pt-224
export OUTPUT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation-real
export LOG_ROOT=/root/autodl-tmp/logs/prompt-ablation-real
```

`PRETRAINED_PATH` 是兼容本分支的 LeRobot PI05 checkpoint 目录，包含 `config.json`、
`model.safetensors`、`policy_preprocessor.json` 和 `policy_postprocessor.json`。
`TOKENIZER_PATH` 应包含 `tokenizer_config.json` 和 `tokenizer.json` 或 `tokenizer.model`。
显式传入该 tokenizer 路径可覆盖 checkpoint 内记录的旧机器路径。
脚本使用 `--policy.type=pi05 --policy.pretrained_path=...`，由真机数据建立输入输出特征，
并在训练时使用当前数据集的归一化统计量。

其他服务器可先设置 `WORK_ROOT=/home/wyn/experiments` 等可写目录：未单独设置的模型、数据、
输出、日志、临时文件和缓存路径都会以它为基准。`WORK_ROOT` 默认 `/root/autodl-tmp`。
`PYTHON` 可指定当前训练环境的 Python；`accelerate` 和 `lerobot-train` 也需在同一环境的 PATH 中。

## 启动示例

先检查四个任务并预览完整命令；不需要 CUDA，不载入模型，也不创建输出目录：

```bash
DRY_RUN=true bash examples/training/train_pi05_real.sh all full_reference 0
```

单卡 GPU 0 训练任务 1；换成 `2`、`3`、`4` 即训练相应任务：

```bash
GPU_IDS=0 BATCH_SIZE=8 bash examples/training/train_pi05_real.sh 1 full_reference 0
```

顺序训练四个任务：

```bash
GPU_IDS=0 BATCH_SIZE=8 bash examples/training/train_pi05_real.sh all full_reference 0
```

两卡训练网球任务（`BATCH_SIZE` 是每卡 batch size）：

```bash
GPU_IDS=0,1 NUM_PROCESSES=2 BATCH_SIZE=8 \
  bash examples/training/train_pi05_real.sh 3 full_reference 0
```

选择消融配置或调整 chunk：

```bash
bash examples/training/train_pi05_real.sh 1 dual_prompt_only 0
CHUNK_SIZE=8 MASKED_STEPS=6 N_ACTION_STEPS=8 \
  bash examples/training/train_pi05_real.sh 1 full_reference 0 --wandb.enable=false
```

可选消融与原入口相同：`full_reference`、`no_bridge`、`direct_dual`、`dual_prompt_only`、
`vlm_only`、`action_only`、`no_cabo`。具体 recipe 见 `train_pi05_prompt_ablation.sh --help`
的说明（也可以不带参数运行原入口查看用法）。

## 默认参数和输出

| 参数 | 默认值 |
| --- | --- |
| `FLOW_STEPS` | 3000，指正式 flow 阶段 |
| `full_reference` 前置阶段 | 750 步 action priming + 250 步 bridge，随后 3000 步 flow |
| `BATCH_SIZE` / `SAVE_FREQ` | 每卡 8 / flow 每 500 步保存，最后一步也保存 |
| `SAVE_STEPS` | 默认未设置；JSON 列表覆盖 `SAVE_FREQ`，例如 `[6000,9000,12000]`，最终步始终保存 |
| `KEEP_PRETRAIN_CHECKPOINT` | 原入口默认 `true`；AutoDL 入口为 `false`，任务成功后清理前置临时权重 |
| `CHUNK_SIZE` / `MASKED_STEPS` / `N_ACTION_STEPS` | 50 / 40 / 50；改变 chunk 时默认 mask 数为约 80% |
| `DTYPE` / `MIXED_PRECISION` | `float32` / `no`，沿用本分支配置 |
| `VIDEO_BACKEND` | `pyav` |
| `COMPILE_MODEL` / `GRADIENT_CHECKPOINTING` | `true` / `true` |
| `NORMALIZATION_MODE` | `QUANTILES`，VISUAL 固定为 `IDENTITY` |

每个任务的名称包含子目录、variant 和 seed。例如任务 1 的最终 flow checkpoint：

```text
$OUTPUT_ROOT/pi05-may-1-put_the_apple_on_the_yellow_plate-full_reference-seed0/checkpoints/003000/pretrained_model
```

日志位于 `$LOG_ROOT/<相同运行名称>.log`。前置阶段保存到同级的
`<运行名称>_next_action_pretrain/checkpoints/001000/pretrained_model`。
每个任务成功训练后，运行目录还会保存 `dataset_info.json`，供真机部署的 `--dataset_info` 使用。
重复运行时如果输出目录已存在，脚本会停止；使用新的 `OUTPUT_ROOT` 或 seed 开启独立实验。
该入口用于新训练，断点恢复应使用 checkpoint 的训练配置和原生 `lerobot-train --resume=true` 流程。

## 缺少 q01/q99 时

v3.0 格式不保证已经有分位数统计量。默认 π0.5 需要 `observation.state` 和 `action` 的
`q01/q99`，缺少时脚本会在加载模型之前报出具体任务和字段，不会静默更改归一化方式。
可以先用仓库工具计算统计量（此命令会更新指定数据集的 `meta/stats.json`，不上传数据）：

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats \
  --repo-id=may-pick-and-place/1-put_the_apple_on_the_yellow_plate \
  --root="$DATASET_BASE/1-put_the_apple_on_the_yellow_plate" --overwrite --no-push-to-hub
```

其他任务同样处理。如果实验明确采用其他归一化，可设置 `NORMALIZATION_MODE=MEAN_STD`
或 `MIN_MAX`；对应统计量也必须存在，这会改变实验配置。

预检检查目录、JSON 元信息、必要文件和统计量；实际启动还检查可见 GPU 和基本 CUDA 运算。
`DRY_RUN` 不解码视频、不读取全部 Parquet 行、不执行模型前向。首次可用较短配置验证真实数据读取：

```bash
FLOW_STEPS=2 SAVE_FREQ=1 BATCH_SIZE=1 COMPILE_MODEL=false \
OUTPUT_ROOT=/root/autodl-tmp/chkpt/pi05-real-smoke \
  bash examples/training/train_pi05_real.sh 1 dual_prompt_only 0 --num_workers=0
```

短测完成后用独立输出目录启动正式配置；短测不代表机器人任务成功率。

## Piper 真机测试

`piper` 分支基于 `prompt-ablation`，提供根目录部署入口 `deploy_piper_vlaa.py`。
训练后复制完整 `pretrained_model/`、对应任务的 `meta/info.json` 和本机 tokenizer，
按 [Piper 部署说明](../piper/README.md) 检查环境并运行。部署使用训练时保存的处理器和归一化统计，
不要把旧版 PI05 模型或配置文件覆盖到该分支。
