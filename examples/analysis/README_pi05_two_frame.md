# 两个样本的顺序对照：E2H 与 H2E

本实验从 `pi05_libero_base` 开始，训练 **2 个 VLM prompt token 和完整 action expert**。
视觉/VLM 主干冻结；不添加 action prompt。一个样本指一个观测帧及其连续
50 步动作目标，不是一条完整视频，也不是一个单独的动作。

| 组别 | 更新 1–3000 | 更新 3001–6000 | 总更新次数 |
| --- | --- | --- | --- |
| E2H | 只用简单样本 | 只用困难样本 | 6000 |
| H2E | 只用困难样本 | 只用简单样本 | 6000 |

各组均为两张 GPU、每卡 batch32。每次更新的 64 个位置重复当前阶段的同一个观测，
但各自使用独立生成的 flow noise/time。每个样本使用 192,000 次，总计 384,000 次；
**不同的观测起点始终只有 2 个**。重复使用同一帧也有不同的 occurrence ID。

两组共享 P1 初始化、同一个样本使用实例池、E1/T1/Bocc 噪声配对、处理器和训练配置。
交换阶段时，样本使用实例携带自己的 ε/t 一起移动。因此这次只交换顺序，
不同时改变曝光比例或帧与噪声的配对。D0 表示候选样本来自此前的 D0 池；
新的实际训练顺序是两阶段计划，不再是此前完整数据集的 W0 顺序。

优化器和学习率按同一条 6000 步时间轴连续运行，切换样本时不重启。
沿用已有学习率配置，所以顺序效应包含样本与学习率阶段、优化器状态的相互作用。
数据归一化继续使用已有 LIBERO/source 处理器，不用这两帧重新估计统计量。

## 1. 复用评分，准备同一对样本

默认在 task 0 的第一条符合条件的 episode 内选样：候选来自原 D0 池，
分别尽量接近该 episode 候选评分的 20% 和 80% 分位；要求简单分数严格低于困难分数，
两个 50 步动作窗口完整、无 padding 且互不重叠。这是**评分分位**，不是视频时间的 20%/80%。

难度仍由此前缓存的 FrameSkip 风格评分代理定义，包含动作变化、视觉变化和夹爪事件信息。
高分不必然等于模型的高预测误差。因此同时记录初始化时两帧的固定 flow loss，供检查这个区别。
不根据训练后的成功率挑样本。默认同一 task/episode 可减少任务、场景差异；
若没有符合条件的样本，程序明确报错，可指定其他 task 或 episode。

下面只读缓存、准备计划，**不重新运行离线评分，也不加载 GPU 模型**。
先运行一次，完成后再启动两个训练终端：

```bash
cd /home/wyn/VLAA/PromptVLA

PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_two_frame prepare \
  --source-pair /data1/wyn/prompt-work/source-pair-r1 \
  --scores /data1/wyn/prompt-work/frameskip-r1/scores.npz \
  --output /data1/wyn/prompt-work/twoframe-r1 \
  --task-index 0 --steps 6000
```

输出 `selection.easy`、`selection.hard`，包含 dataset index、task、episode、frame index、
评分及分位。生成 `pair.json`、`E2H.npz`、`H2E.npz`，并校验评分、数据和来源。
也可以增加 `--episode-index N`；手动指定时同时提供 `--easy-index I --hard-index J`，
这里 I/J 是整个数据集的全局帧索引，仍需满足同 episode、完整且不重叠等条件。
更换样本或步数时使用新的 `--output` 和对应 `PAIR_DIR`，不覆盖已有计划。

## 2. 四张 GPU 并行运行两组

终端一，GPU 0、1，简单到难：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
PAIR_DIR=/data1/wyn/prompt-work/twoframe-r1 \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29586 FLOW_STEPS=6000 REPEAT_ID=1 \
TRAIN_ACTION_EXPERT=true DRY_RUN=false \
bash examples/training/train_pi05_two_frame.sh E2H \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

终端二，GPU 2、3，难到简单：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
PAIR_DIR=/data1/wyn/prompt-work/twoframe-r1 \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29587 FLOW_STEPS=6000 REPEAT_ID=1 \
TRAIN_ACTION_EXPERT=true DRY_RUN=false \
bash examples/training/train_pi05_two_frame.sh H2E \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

先设 `DRY_RUN=true` 可以查看实际命令。入口固定两卡×batch32、bf16、关闭 compile、
训练完整动作网络、无 loss 加权、E1/T1/Bocc 和仅保存最后的完整 checkpoint；
仅接受额外 W&B 参数，防止两组配置意外分叉。

模型名自动带入真实样本索引（下例的 I/J 是占位符，不需手动输入）：

```text
pi05-twoframe-r1-E2H-eI-hJ-steps6000-aexpert-base-vlm2-P1-D0-E1-T1
pi05-twoframe-r1-H2E-eI-hJ-steps6000-aexpert-base-vlm2-P1-D0-E1-T1
```

默认保存在 `/data1/wyn/chkpt/2601-lerobot/PromptVLA/`，完整 checkpoint 只有
`checkpoints/006000/pretrained_model/`。不生成中间的完整模型；保留小体积的诊断与审计文件。

## 3. 先比较两帧各自的学习和遗忘

两组的探针始终是**同一对训练帧**，固定顺序 sample0=easy、sample1=hard。
诊断使用同一份独立缓存的探针噪声、t=[0.2,0.5,0.8]，不消耗训练 RNG。
在更新 0、50、100、每 250 步（含切换点 3000）及最终 6000 记录：

- 每帧有效动作坐标上的 flow MSE；
- 两帧 MSE 的等权平均；
- 沿用的 prompt 有效秩诊断。

W&B 指标 `fixed_training_probe_sample0_mse`、`fixed_training_probe_sample1_mse`
分别对应简单帧、困难帧。每组训练完成后自动审计并生成 `two_frame_losses.csv`，
列为 `step,easy_mse,hard_mse,balanced_mse`。原始记录在
`prompt_rank/fixed_training_probe_loss.jsonl`，固定帧信息在 `prompt_rank/manifest.json`。

两个终端都完成后，再做配对审计：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_two_frame audit \
  /data1/wyn/prompt-work/twoframe-r1 \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-twoframe-r1-*-e*-h*-steps6000-aexpert-base-vlm2-P1-D0-E1-T1
```

期望 `paired_order_only_passed`。审计检查全部实际训练步骤，验证只有两个观测起点、
各阶段样本与曝光次数、同一个实例/噪声库、初始化、来源、训练配置、代码和固定探针；
不宣称完整浮点数值确定性。一次只审计一对 E2H/H2E。

比较终点两帧的 `balanced_mse`，并检查每帧从 3000 到 6000 的变化。
若第二阶段样本误差下降、第一阶段样本误差上升，支持出现顺序遗忘。
这些是训练样本上的固定噪声诊断，不是独立验证集分数；不能仅由一对样本确定通用课程优劣。

## 4. 可选：测试 LIBERO 四个 suite

```bash
GPU_ID=0 FLOW_STEPS=6000 REPEAT_ID=1 \
bash run_eval_libero_two_frame.sh E2H all

GPU_ID=2 FLOW_STEPS=6000 REPEAT_ID=1 \
bash run_eval_libero_two_frame.sh H2E all
```

可在两个终端并行。默认 `EVAL_SEED=1000`、`EPISODES_PER_TASK=10`，每组 400 episodes，
保留单行进度与最终成功率。入口自动查找含真实样本索引的模型名，校验来源和训练范围，
使用完整训练后的动作网络。若准备目录不同，给训练和测试都传相同的 `PAIR_DIR`。

两样本训练的主要目的，是检验这对样本上的顺序敏感性与遗忘。
四个 suite 的成功率可补充观察泛化，但不应预期只训练两帧能维持全数据训练的成功率。
