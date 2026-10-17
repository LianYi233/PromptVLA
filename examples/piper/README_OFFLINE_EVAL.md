# Piper 训练样本上的离线动作验证

`eval_piper_offline.py` 读取 May-pick-and-place 单个任务的 LeRobot v3.0 数据，
将当前时刻两路 RGB、机器人 state 和数据集中保存的 task 文本输入 PI05，比较预测 action
与同一时刻开始的真实 action 序列。只使用录制的视频，不打开相机、CAN 或 Piper SDK。

训练日志的 `loss≈0.3` 是带随机噪声/时间的 flow 速度预测 MSE，不能直接解释为
关节角度误差，也不能单凭这个数值认定训练无效。这里运行完整 flow 采样，输出反归一化后
的动作误差；它检查训练样本拟合程度，不测真机闭环成功率或未见数据泛化。

## 运行

在与训练相同的 conda 环境、完整 `piper` checkout 根目录运行。需要原始的 `data/`、
`meta/`、`videos/` 和完整 checkpoint；只给视频或 `info.json` 无法获得对齐标签。
无需安装机械臂驱动。初次建议用默认 256 个观测快速检查，再扩大到全部帧。

```bash
cd /root/lerobot060
git pull --ff-only origin piper

# 改为你实际的运行组目录。任务 3 的示例：
PIPER_RUN_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/piper/piper-a100-retrain-01
PIPER_TASK=3-move_the_tennis_from_yellow_plate_to_blue_plate
PIPER_RUN_NAME=pi05-may-${PIPER_TASK}-full_reference-seed0

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python eval_piper_offline.py \
  --policy_path "${PIPER_RUN_ROOT}/${PIPER_RUN_NAME}/checkpoints/012000/pretrained_model" \
  --dataset_root "/root/autodl-tmp/datasets/May-pick-and-place/${PIPER_TASK}" \
  --tokenizer_path /root/autodl-tmp/models/google/paligemma-3b-pt-224 \
  --episodes all --stride 8 --max_samples 256 --execution_steps 8 --seed 0 \
  --output_dir "/root/autodl-tmp/eval/piper/${PIPER_RUN_NAME}-012000"
```

脚本拒绝覆盖已有结果目录；再次评估时换一个 `--output_dir`。
其他机器将 checkpoint、dataset、tokenizer、output 四个路径改为本机路径即可。
`--tokenizer_path` 指向包含 `tokenizer_config.json` 的目录，覆盖训练机器的旧绝对路径。
默认保留 checkpoint dtype、AMP 和 flow 推理次数，关闭 compile/RTC；每次只推理一个观测。
如需与部署一致，可显式传 `--num_inference_steps 10`。

四个任务各自对应一个独立模型；逐个修改 `PIPER_TASK` 和模型路径：

| 任务 | 目录名 |
| --- | --- |
| 1 | `1-put_the_apple_on_the_yellow_plate` |
| 2 | `2-remove_the_cuboid_from_blue_plate` |
| 3 | `3-move_the_tennis_from_yellow_plate_to_blue_plate` |
| 4 | `4-pick_the_red_cube_into_the_yellow_plate` |

更完整的检查可以将参数改成：

```bash
--episodes 0,1,2,3,4 --stride 1 --max_samples 0
```

这会检查前五个 episode 的每一帧。`--episodes all --stride 1 --max_samples 0`
检查所有训练 episode 的所有帧，需要更多时间。若 checkpoint 的 `train_config.json`
指定了训练 episode 子集，`all` 只选该子集；显式指定非训练 episode 会报错。
默认先按 episode 内 `frame_index % stride == 0` 选择候选，再按时间顺序均匀选取至多
`max_samples` 个观测。实际 episode/frame/index 全部写入结果，未自动挑选低误差样本。

## 本地机器一次生成四个任务的对比图

```bash
cd /home/drx/DJC/lerobot-piper
git pull --ff-only origin piper
bash examples/piper/eval_four_tasks.sh
```

默认按任务 1 → 2 → 3 → 4 顺序评估，每个任务最多评估 256 个观测，均匀选取 6 个观测
绘图（样本充足时共 24 张）。不会按误差大小筛图。每张图仍表示某个录制观测时刻开始的
动作预测窗口，不是完整 episode 的闭环轨迹。评估样本不足时，图片数以实际样本数为限。

脚本内置本次提供的路径，任务 3 不添加 `012000` 子目录：

| 项目 | 默认路径 |
| --- | --- |
| 数据集父目录 | `/media/drx/a1ea95d8-943f-4c65-b4a3-05f5927573b7/dataset/May-pick-and-place` |
| 任务 1、2、3 模型父目录 | `/media/drx/a1ea95d8-943f-4c65-b4a3-05f5927573b7/Wu_Yinan/Priming` |
| 任务 1、2 模型相对路径 | `pi05-may-<任务目录名>-full_reference-seed0/012000/pretrained_model` |
| 任务 3 模型相对路径 | `pi05-may-3-move_the_tennis_from_yellow_plate_to_blue_plate-full_reference-seed0/pretrained_model` |
| 任务 4 模型 | `/home/drx/Downloads/wyn/pi05-may-4-pick_the_red_cube_into_the_yellow_plate-full_reference-seed0/012000/pretrained_model` |
| Tokenizer | `/home/drx/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c` |

输出在仓库的 `outputs/piper-offline/four-tasks-<时间戳>/<任务目录名>/`，包含 PNG、CSV、
NPZ 和 summary.json；每个任务的运行日志在时间戳目录下的 `<任务目录名>.log`。
启动前会检查所有选中任务的基础文件路径，详细权重和元数据校验由评估器执行。

```bash
# 仅打印四个命令；不检查本机文件、不加载模型。
bash examples/piper/eval_four_tasks.sh --dry_run

# 每个任务 10 张图，共最多 40 张；评估样本数仍为 256。
PLOTS_PER_TASK=10 bash examples/piper/eval_four_tasks.sh

# 只跑任务 3；输出位置可自行指定（不得已有同名任务结果目录）。
PIPER_OUTPUT_ROOT=/home/drx/piper-eval-task3 bash examples/piper/eval_four_tasks.sh 3
```

`PIPER_POLICY_1` 至 `PIPER_POLICY_4` 可分别覆盖模型路径；`PIPER_DATASET_BASE`、
`PIPER_MODEL_BASE`、`PIPER_TOKENIZER_PATH` 可覆盖基础目录。更多采样参数见脚本 `--help`。

## 输出及含义

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 配置、数据标识、有效计数、前 1/8/50 步误差、各 episode 的前 8 步误差 |
| `metrics.csv` | 各关节/夹爪的 MAE、RMSE、有符号偏差、95% 绝对误差分位数和最大误差 |
| `actions.csv` | 每个有效 anchor/offset 对应的预测、标签、保位基线及精确帧编号 |
| `predictions.npz` | 完整 `[N,H,7]` 预测/标签、有效掩码、基线、任务文本和索引，可自行重画 |
| `episode_*_frame_*.png` | 选中观测的相机 RGB，以及 7 个动作维度的 `teleoperation` / `action from model` 对比曲线 |

图中黑线 `teleoperation` 是数据集记录的 action 标签，蓝线 `action from model` 是模型
反归一化后的预测。图中只显示这两条曲线；保位基线仍保留在数值结果中。

终端直接打印前 1 步、实际执行的前 8 步和完整 chunk 的关节 MAE（度），
以及夹爪 MAE（数据集原始单位），并列显示 `hold_current_state` 基线。
夹爪误差不与关节误差直接混为一个平均数。逐关节 CSV 保留数据集弧度单位，
角度换算基于本项目 Piper 驱动的 rad 约定；若数据并非该单位，不应解释度数换算结果。
本项目驱动通常将夹爪开度记为约 0..1，脚本不会擅自改标签单位或重新标零。

`hold_current_state` 将当前 state 按 action 的名称顺序重排，整个预测段保持不变。
对于短时间预测，保位本身可能已有很小误差，所以不能只看模型的绝对 MAE：

- 先看前 8 步是否在各关节和夹爪上优于保位，再检查完整 chunk 的走势。
- 结合图像确认任务/视角正确，观察动作曲线是否有固定偏移、反向、尺度不符，
  以及夹爪闭合/张开的时机是否偏离标签。
- 若训练样本上的预测明显不好，先核对 checkpoint、任务、统计量、动作单位与顺序，
  然后比较不同训练步数；不能只依靠降低训练 loss 判断。
- 若离线拟合较好而实机抓取差，需要进一步检查部署视角、初始状态、动作跟踪和推理间隔。
  离线每个锚点都使用真实录制观测，不会暴露闭环误差累积。

可以用完全相同参数分别检查 `006000`、`009000` 和 `012000`，输出到不同目录。
相同 `seed` 和绝对样本 index 会生成相同初始噪声，便于公平比较；每个观测只采样一次，
不会从多个预测中挑选与标签最接近的结果。需要估计采样波动时，用不同 seed 分别重跑。
图和数值均不预设“好”的阈值，夹爪尺寸、物体大小和关节姿态都会影响可接受误差。

## 对齐与归一化

使用分支自带 `LeRobotDataset` 读取合并视频的 episode 时间偏移，按数据集 fps 构造
`action_delta_indices=[0,1,...,H-1]`；第一个预测对应当前记录的 action，不擅自错后一帧。
每个样本都会核对当前 action 与原始 parquet 标签，以及 padding mask 与 episode 边界。
episode 末尾补齐的标签不计入误差，也不会接到下一条 episode。

输入只包含当前图像、state、task，action 标签和未来状态不会送入模型。
RGB 使用训练读取路径的 uint8 -> float/255，禁用随机图像增强。
使用 checkpoint 保存的 pre/postprocessor 与统计文件，不在评估样本上重算统计量，
也不裁剪预测动作以掩盖大误差。结果按有效“观测锚点/预测偏移”对等权统计；
相邻 chunk 重叠时，同一绝对标签可能被不同锚点预测，这是多步预测误差，不是独立轨迹数。

运行失败不会生成标记成功的 `summary.json`。默认生成 3 张图；无 matplotlib 时可先用
`--plots 0`，或在训练环境安装 matplotlib。全部图都由实际录制观测和实际推理结果生成。
