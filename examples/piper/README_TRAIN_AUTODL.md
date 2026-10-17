# AutoDL：重新训练 Piper 四个真机任务

之前的入口是 `examples/training/train_pi05_real.sh` 和 `train_pi05_real.py`，已经支持
`May-pick-and-place` 的 LeRobot v3.0 数据。新增的 `train_piper_autodl.sh` 复用此入口，
集中设置 AutoDL 路径和本次重训参数。训练只使用离线数据，不连接机械臂或相机。

抓取失败、loss 停滞或需要检查小样本拟合时，见
[训练诊断与可选动作映射训练](README_TRAIN_DIAGNOSIS.md)。该入口不会改变本文的默认方法。
训练期间的 W&B、动作预测误差和参数更新量配置见
[动作指标说明](README_ACTION_METRICS.md)。首次在线运行需执行 `wandb login`，
离线运行可设置 `WANDB_MODE=offline`。

## 1. 获取代码和准备环境

在 AutoDL 现有仓库中切换到 `piper`（如有未提交修改，先自行提交或暂存）：

```bash
cd /root/lerobot060
git fetch origin
git switch piper
git pull --ff-only origin piper
```

如果已有该目录且当前分支为 `piper`，使用 `git pull --ff-only origin piper` 更新。
激活之前训练使用的 Python >=3.12 环境，例如 `conda activate lerobot`。首次在这个
checkout 安装训练依赖：

```bash
python -m pip install -e '.[training,pi]'
```

训练入口会将当前 checkout 的 `src/` 放在 `PYTHONPATH` 首位。
`python`、`accelerate` 和 `lerobot-train` 应来自同一个训练环境；可用
`which python accelerate lerobot-train` 核对。无需安装 Piper SDK、CAN 或 RealSense 驱动。

## 2. 设置路径

```bash
export DATASET_BASE=/root/autodl-tmp/datasets/May-pick-and-place
export PRETRAINED_PATH=/root/autodl-tmp/models/pi05_libero_base
export TOKENIZER_PATH=/root/autodl-tmp/models/google/paligemma-3b-pt-224
export RUN_GROUP=piper-a100-12k-01
```

以上是当前 AutoDL 默认路径；如果模型位置不同，按实际目录修改。
预训练目录需含完整权重、配置及 pre/postprocessor；tokenizer 必须在本地完整保存。
入口默认离线加载，避免训练服务器旧路径或 Hub 网络连接问题；不会自动下载缺失文件。

| TASK | 子目录 | 任务 |
| --- | --- | --- |
| `1` | `1-put_the_apple_on_the_yellow_plate` | 苹果放到黄盘 |
| `2` | `2-remove_the_cuboid_from_blue_plate` | 从蓝盘移走长方体 |
| `3` | `3-move_the_tennis_from_yellow_plate_to_blue_plate` | 网球从黄盘移到蓝盘 |
| `4` | `4-pick_the_red_cube_into_the_yellow_plate` | 红方块放到黄盘 |

每个目录必须有 `data/`、`meta/`、`videos/`。预检会检查 v3.0 元信息、任务/episode
Parquet 文件、各相机视频是否存在，以及 state/action 的归一化统计量。
语言指令使用数据内的 `task_index` 和 `meta/tasks.parquet`；不从目录名重新生成。
动作单位、关节顺序及相机 key 保留数据定义，不自动转换。

## 3. 预检和正式训练

预检四个任务，不加载模型、不占 GPU、不写训练输出：

```bash
DRY_RUN=true bash examples/training/train_piper_autodl.sh all
```

正式训练，默认使用一张 A100（GPU 0）：

```bash
bash examples/training/train_piper_autodl.sh all
```

`all` 按 1→2→3→4 顺序训练四个独立模型，每个任务都从同一个 `PRETRAINED_PATH`
开始；不会将前一任务的权重传给后一任务。所有任务的路径预检通过后才启动第一个训练。
某个任务报错时停止整个队列，保留已有日志和 checkpoint。

只训练一个任务，或将每卡 batch 改为 4：

```bash
# 单卡重训网球任务
bash examples/training/train_piper_autodl.sh 3

# 单卡顺序训练四个任务，自动使用一个进程
GPU_IDS=0 BATCH_SIZE=4 bash examples/training/train_piper_autodl.sh all
```

正式运行前如需验证真实视频读取和前后向，可用独立目录做两步测试：

```bash
RUN_GROUP=piper-smoke-01 GPU_IDS=0 BATCH_SIZE=1 FLOW_STEPS=2 SAVE_STEPS='[2]' \
COMPILE_MODEL=false NUM_WORKERS=0 \
  bash examples/training/train_piper_autodl.sh 1 dual_prompt_only 0
```

此短测选择 `dual_prompt_only`，避免仍然执行完整的 1000 步前置训练；正式实验使用默认
`full_reference`。短测通过只说明所测任务的数据读取与训练流程可运行。

## 4. 本次默认配置

| 配置 | 默认值 |
| --- | --- |
| recipe / seed | `full_reference` / `0` |
| GPU / 每卡 batch | `0` / `8`（默认全局 batch 为 8） |
| 精度 | `DTYPE=float32`、`MIXED_PRECISION=no` |
| 可训练参数 | VLM prompt 16 tokens + action prompt 16 tokens；骨干和投影层冻结 |
| CABO | 启用，`CABO_RATIO=2.0` |
| 前置训练 | 总计 1000 步，包含 750 步 action priming + 250 步 bridge |
| 正式 flow | `FLOW_STEPS=12000`；连同前置阶段，每个任务共 13000 次更新 |
| 保存 | `SAVE_STEPS='[6000,9000,12000]'`，只保留这三个 flow checkpoint |
| 前置临时权重 | 阶段衔接需暂存 001000；该任务成功且最终模型存在后自动清理 |
| 训练动作段 / mask | 50 / 40 |
| 部署默认执行段 | `N_ACTION_STEPS=8`，每次预测后仅执行前 8 步再读取观测 |
| 归一化 / 视频 | 数据集自身 `QUANTILES` / `pyav` |
| compile / gradient checkpointing | `true` / `true` |
| NTK 额外快照 | 默认关闭 |
| W&B | 默认开启，项目 `piper-action-fit`；不上传模型 artifact |
| 固定观测动作评估 | 每 500 步，每个划分 16 个观测；开始、阶段边界和结束也记录 |

`N_ACTION_STEPS=8` 只控制 checkpoint 的部署默认执行长度，不把训练目标从 50 步改为
8 步；若需复现原来执行 50 步的配置，可显式设置 `N_ACTION_STEPS=50`。
训练读取数据集 fps，部署也读取相同元信息；不要通过修改训练 fps 来降低机械臂速度。

可用环境变量覆盖参数，例如 `FLOW_STEPS=3000 SAVE_STEPS='[3000]'`、`BATCH_SIZE=4`、
`DTYPE=bfloat16`（自动配套 `MIXED_PRECISION=bf16`）。不同精度或 batch 会改变实验配置。
支持原有消融，例如 `bash examples/training/train_piper_autodl.sh all no_bridge 0`。

`SAVE_STEPS` 优先于 `SAVE_FREQ`，最终一步始终保存；`SAVE_STEPS='[]'` 表示只存最终模型。
默认不会保存 3000 步或 NTK 的 0/750 步快照。修改 `FLOW_STEPS` 时同时修改 `SAVE_STEPS`，
保存步数超过总步数会在预检时报错。前置训练有独立的保存计划，不继承 flow 的保存步数。
如需保留前置权重，设置 `KEEP_PRETRAIN_CHECKPOINT=true`。任务失败时自动保留临时权重；
清理只作用于本次新建任务，已有运行目录仍拒绝覆盖。`checkpoints/last` 是指向最终保存点的
符号链接，不是另一份模型。更新代码不会改变已运行进程的步数或保存策略。

## 5. 输出、日志和传回真机

在实机部署前，可先用训练样本离线比较预测动作与标签，见
[离线动作验证](README_OFFLINE_EVAL.md)。该检查不连接机械臂，可比较 6000/9000/12000 步模型。

默认输出根目录：

```text
/root/autodl-tmp/chkpt/2601-lerobot/piper/<RUN_GROUP>/
```

每个任务有独立目录。以网球任务为例：

```text
pi05-may-3-move_the_tennis_from_yellow_plate_to_blue_plate-full_reference-seed0/
  dataset_info.json
  checkpoints/006000/pretrained_model/
  checkpoints/009000/pretrained_model/
  checkpoints/012000/pretrained_model/
```

训练成功后自动将该任务的 `meta/info.json` 复制为 `dataset_info.json`，便于与模型一起
传回真机。部署需要完整 `pretrained_model/`，包括模型、配置、处理器及其引用的统计量文件。
使用正式 flow 目录下的 checkpoint，不使用 `_next_action_pretrain` 目录部署。
训练保存完整模型，四个任务的多个 checkpoint 会占用较多空间，请将输出放在数据盘。

日志目录为 `/root/autodl-tmp/logs/piper/<RUN_GROUP>/`，每个任务一个 `.log`。
可自行设置 `OUTPUT_ROOT` 和 `LOG_ROOT` 覆盖默认路径。
未指定 `RUN_GROUP` 时使用启动时间；建议像上面一样显式设置，保证预检与正式运行路径一致。
重训同一任务时换一个组名；已有输出目录会报错，不会覆盖。中断续训需使用已有 checkpoint
和原生 resume 流程，不能把已训练目录当作全新实验继续启动。

传回真机后，`--dataset_info` 指向该任务的 `dataset_info.json`，
`--tokenizer_path` 指向真机上的 tokenizer 目录。其他检查见 [部署说明](README.md)。
离线重训本身不能证明之前的碰盘子和驱动故障已解决，部署仍保留通信及故障检查。

## 缺少归一化统计量

若预检报 `missing action.q01` 等错误，先按
[原训练说明](../training/README_pi05_real.md#缺少-q01q99-时) 对相应数据集补充分位数统计量，
然后重新预检。脚本不会自动修改数据或改用其他归一化。
