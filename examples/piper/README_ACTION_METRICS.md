# Piper 训练期间的动作误差与 W&B 曲线

`train_piper_autodl.sh` 和 `train_piper_fit.sh` 现在默认启用 W&B，项目名
`piper-action-fit`。原始通用训练入口仍是显式开启诊断，其他机器人不受 Piper 预设影响。
只上传配置和标量指标，默认 `wandb.disable_artifact=true`，不上传模型权重或相机视频。
已有训练进程不会因 `git pull` 自动获得这些曲线；新启动或使用新代码恢复的训练才会记录。

## 启动

```bash
cd /root/lerobot060
git pull --ff-only origin piper
wandb login

DRY_RUN=false RUN_GROUP=piper-fit-task4-metrics-01 \
  bash examples/training/train_piper_fit.sh 4 projections 0
```

这仍是一个 episode 的拟合检查：750 步 priming、250 步 bridge、2,000 步 flow。
W&B 会出现两个 run：带 `_next_action_pretrain` 后缀的 priming/bridge run，以及正式 flow run。
每个 run 的横轴是本阶段的更新步数；阶段结束会关闭并刷新 run，避免复用同一个 run 时步数回退。
前一个 run 的 750 步为 priming 结束，1,000 步为 bridge 结束。

不能联网时设置 `WANDB_MODE=offline`，此时不会实时同步网页：

```bash
WANDB_MODE=offline RUN_GROUP=piper-fit-task4-metrics-offline \
  bash examples/training/train_piper_fit.sh 4 projections 0
```

网络恢复后对日志目录中的 `wandb/offline-run-*` 执行 `wandb sync <目录>`。
`WANDB_ENABLE=false` 则完全不初始化 W&B，本地 JSONL 指标仍保留。

## 记录的指标

这里的动作评估做完整的 flow 去噪推理，输入只有当前双相机图像、当前 state 和 task。
标签单独缓存，既不输入模型，也不使用示范的未来 state。预测经过 checkpoint 的动作
反归一化后与数据标签逐时刻比较；action offset 0 对齐当前 frame，屏蔽 episode 尾部补齐位置。

默认对每个数据划分固定选取 16 个均匀分布的观测，禁用图像增强，固定每个观测的初始噪声。
在阶段开始、每 500 步、priming/bridge 边界和阶段最终步评估。评估不会多存 checkpoint。
临时切换到完整观测条件的 flow 推理，结束后恢复训练阶段、训练模式和随机数状态。
所以 priming 阶段的动作误差也衡量完整条件预测器，不是 action-only inpainting 的训练 loss。

W&B 中优先查看下面这些曲线（`h8` 表示一个预测 chunk 的前 8 步）：

| 曲线 | 含义 |
| --- | --- |
| `train/loss` | 训练目标的 loss，不能直接解释为动作角度误差 |
| `train/loss_per_dim/0` … `/6` | 按 action 顺序展开的逐维训练 loss |
| `eval/action_train/h8/joint_mae_deg` | 六关节平均绝对误差，单位为度 |
| `eval/action_train/h8/joint_rmse_deg` | 六关节 RMSE，对较大误差更敏感 |
| `eval/action_train/h8/joint_p95_deg` | 所有有效关节绝对误差的 95% 分位数 |
| `eval/action_train/h8/joint_2_mae_deg` | 单个关节 MAE，六关节均记录 |
| `eval/action_train/h8/gripper_mae` | 夹爪 MAE，保留数据集单位，不与角度混合平均 |
| `eval/action_train/h8/joint_mae_deg_ratio_to_hold` | 模型 MAE / 保持当前关节位置的 MAE；小于 1 表示优于该基线 |
| `eval/action_train/h8/moving_joint_mae_deg` | 只统计示范目标至少有一个关节距当前 state 超过 1° 的动作 |
| `eval/action_train/h8/joint_delta_mae_deg_per_step` | chunk 内相邻动作增量的误差，辅助观察动作变化是否被学到 |
| `train/action_update/prompt_tokens/grad_rms` | action prompt 在抽样更新步上的梯度 RMS |
| `train/action_update/prompt_tokens/relative_update` | 实际更新的 L2 范数 / 更新前参数 L2 范数 |
| `train/action_update/action_out_proj/update_rms` | 动作输出层实际参数变化；默认冻结时应为 0 |
| `train/cabo/action_prompt_scale` | CABO 对 action prompt 的学习率缩放 |
| `train/cabo/effective_action_prompt_lr` | CABO 作用后的 action prompt 学习率 |
| `train/optimizer_step/skipped` | 日志窗口内被跳过的更新比例 |

同时记录 `h1` 和整个 chunk（默认 `h50`），以及夹爪 RMSE/增量误差/保持基线。
静止基线误差为 0 时不写无意义的比值；没有运动目标或有效相邻动作时，相应曲线缺点，
并写出 `moving_pairs`、`adjacent_pairs` 等样本数，而不是填成 0。

参数监测涵盖 `vlm_prompt_tokens`、`prompt_tokens`、`action_in_proj`、`action_out_proj`。
默认每 50 个更新抽样一次，记录真实 optimizer.step 前后的差值，包含 CABO 和 weight decay
的作用。每个日志窗口写入这些抽样值的均值；这不是整个窗口累计位移，也不等同于 CABO
内部排除 weight decay 后估计的相对学习更新量。冻结层或没有梯度的 prompt 更新为 0 是预期行为。
大 action transformer 仍冻结，以上指标不会把被冻结的 action head 误报成正在训练。

## 训练拟合与泛化分开看

默认 `EVAL_SPLIT=0` 保留全部数据用于训练，只有 `action_train` 曲线，不能称为验证集误差。
在全数据实验中可留出 10% episode，再记录同样的 `eval/action_val/...` 曲线：

```bash
FIT_EPISODES=all EVAL_SPLIT=0.1 FLOW_STEPS=12000 \
  SAVE_STEPS='[6000,9000,12000]' RUN_GROUP=piper-task4-validation-01 \
  bash examples/training/train_piper_fit.sh 4 projections 0
```

使用数据集已有的按任务、episode 划分逻辑：每个任务末尾约 10% episode 留出，不参与更新。
这改变了训练数据量，结果不能当作原来全量训练的同一设置。一条 episode 的拟合实验保持
`EVAL_SPLIT=0`，否则会没有训练 episode。所有指标仍是离线、给定示范观测的预测误差，
不是闭环任务成功率，也不是由模型自身动作产生后续观测的 rollout。

判断时结合：动作误差持续下降、运动片段误差下降、优于静止基线、夹爪误差改善，以及
留出集趋势。仅有梯度/更新不为 0 不能证明充分训练；低训练误差而高留出误差提示泛化问题。

## 文件与开销

每个阶段的输出目录包含：

- `training_diagnostics.jsonl`：训练窗口均值、逐维 loss、CABO、抽样更新量及非有限值计数。
- `action_eval.jsonl`：每次动作评估的精确 step 和所有动作误差指标。
- `action_eval_samples.json`：固定观测的 episode/frame/绝对 index、随机种子、推理步数等设置。
- `wandb/`：W&B 本地文件，离线模式也保留。

临时预训练清理只删除 `checkpoints/`，上述诊断文件及 W&B 日志会保留。
默认每次做 16 次预测；有留出集则共 32 次。实际耗时记录在 `eval/action_eval_seconds`。
启用模型编译时第一次预测可能额外等待编译；日志会先打印 `PIPER_ACTION_EVAL starting`。
需要降低开销时，例如设置 `ACTION_EVAL_FREQ=1000 ACTION_EVAL_SAMPLES=8`。

环境变量 `ACTION_EVAL_FREQ`、`ACTION_EVAL_SAMPLES`、`ACTION_EVAL_SEED`、`ACTION_UPDATE_FREQ`
对应 `--piper_eval.freq`、`.samples`、`.seed`、`.update_freq`。
`N_ACTION_STEPS` 同时设置评估的执行窗口。设置 `ACTION_EVAL_FREQ=0 ACTION_UPDATE_FREQ=0`
可关闭额外预测和参数更新采样；基础 loss/CABO 本地日志保持启用。
支持单卡/CPU 和普通 DDP；不支持 FSDP/DeepSpeed 的分片参数。双卡时只有 rank 0 做固定预测，
其他 rank 等待；每步梯度在同步之后采样。W&B 只由 rank 0 写入。
逐维 flow loss 是 rank 0 的窗口均值，`train/loss` 保留训练器原有的跨卡归约；
额外的 `train/policy_rank0/loss` 不会覆盖它。

本次通过 CPU 数值、固定推理隔离、参数更新和脚本入口等 28 项测试；双进程 Gloo
实测被执行环境的网络权限限制而跳过。未使用用户权重/数据完成 GPU 训练或在线 W&B 实测。
