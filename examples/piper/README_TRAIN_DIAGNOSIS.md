# Piper：抓取失败与训练拟合检查

本文基于 `piper` 分支的代码检查，不是对用户本地权重或训练日志的实测报告。
新增的训练期间动作误差与 W&B 曲线见 [动作指标说明](README_ACTION_METRICS.md)。
机械臂曾压到桌面，并报告关节 2 的 `collision_status` 和 `driver_error_status`。
保护停机符合该现象，但仍需要查清模型输出或执行过程为什么会产生接触。
保留现有碰撞、反馈及 CAN 检查；先做离线拟合检查。

## 代码检查结果

- `full_reference` 只训练两组 prompt：`16×2048 + 16×1024 = 49,152` 个参数。
  VLM、动作 transformer、动作输入/输出线性映射都被冻结。旧的
  `freeze_vision_encoder`、`train_expert_only` 参数不会解除这个冻结规则。
  跨到 Piper 的动作空间时，这是一项需要验证的容量限制，还不能认定是根因。
- 默认起点是 `pi05_libero_base`。同为 7 维并不能证明两个数据源的动作语义、单位、
  相机视角一致；本地基础权重的实际来源也需要用户确认。
- 默认优化器峰值学习率是 `2.5e-5`，衰减下限 `1e-5`。CABO ratio=2 会把
  action prompt 的相对更新量限制在当前 VLM prompt 相对更新量的一半以内；
  它不是把 action prompt 学习率乘以 2。是否缩得过小，需要查看实际缩放日志。
- 单卡 batch=8、正式训练 12,000 步对应 96,000 个样本锚点，四个任务分别约
  5.65、10.25、10.21、9.49 次数据遍历，另有预训练阶段。这个步数不自动代表充分收敛。
- 训练入口给处理器覆盖的是当前 Piper 数据集的统计量；加载处理器权重时会保留
  该覆盖。动作 loss 也排除了 episode 尾部补齐的位置。目前未在这两处发现明显错误。
- 控制时每 8 个动作按 10 Hz 发送，随后等待约 0.22 秒推理，仅考虑这两项，
  平均约 7.8 个动作/秒，还不含相机及其他开销。MOVE J 速度参数也影响实际跟踪。
  因此离线误差小仍不等于现场动作能完全复现示范时序。

训练的 flow loss 是随机噪声、随机时间点下的速度预测 MSE；它不是反归一化后
最终动作的关节角误差。`0.3` 不能直接换算成 `0.3 rad`，也没有通用的“必须低于
0.01 才能抓取”阈值。需要同时比较 loss 曲线、动作误差和静止基线。

## 本地诊断日志

PI05 新训练会在每个阶段的输出目录生成 `training_diagnostics.jsonl`，无需 W&B。
按 `log_freq` 写出统计窗口，最后一步也会写出；不增加模型前向或随机采样。

每行包含精确 `step`、`window_updates`、参数数量、动作名称、`train` 指标，以及
`diagnostics_rank0` 中各指标的 mean/min/max/count/nonfinite：

- `loss_per_dim/0` 至 `loss_per_dim/6`：逐维 flow loss，仍在训练目标空间。
- `valid_action_fraction`：有效动作比例；`all_padding_samples`：全部补齐的样本数量。
- `cabo/action_prompt_scale`、两个 `cabo/effective_*_prompt_lr`：实际缩放和学习率。
- `cabo/*relative_update_rate`：两个 prompt 的相对更新量。
- `optimizer_step/skipped`：因非有限梯度等而跳过更新的比例。

关闭 CABO 或 action-only priming 阶段不会凭空产生 CABO 指标。
多卡时 `diagnostics_rank0` 是 rank 0 上现有指标的窗口统计；`train.loss` 保留训练器
原有的分布式归约，不能把所有逐维指标误认为跨卡全局均值。恢复训练会追加行，
从较早 checkpoint 恢复时可能有重复 step，应按对应运行段分析。
临时预训练目录中的 `checkpoints/` 在任务成功后清理；需要保留权重时设置
`KEEP_PRETRAIN_CHECKPOINT=true`。各阶段的诊断文件和 W&B 日志保留在各自输出目录。

## 四个可比较的配置

新增 `examples/training/train_piper_fit.sh`；原 `train_piper_autodl.sh` 默认设置不变。

| PROFILE | 正式 flow 峰值学习率 | CABO | 动作输入/输出层 | 默认模型可训练参数 |
| --- | --- | --- | --- | --- |
| `reference` | 2.5e-5 | 开 | 冻结 | 49,152 |
| `prompt_lr` | 1e-4 | 开 | 冻结 | 49,152 |
| `no_cabo` | 1e-4 | 关 | 冻结 | 49,152 |
| `projections` | 1e-4 | 关 | 训练 | 115,744 |

依次比较相邻配置可分别测试学习率、CABO、动作映射的影响。直接比较 `reference`
和 `projections` 改变了多个因素，只能评价组合配置，不能归因于单个因素。
原训练器固定了预训练阶段的优化器设置，该阶段的峰值学习率仍是 `2.5e-5`；
表中的学习率修改只作用于正式 flow 阶段。

`projections` 显式开启 `--policy.train_action_projections=true`，新增训练的是现有的
`action_in_proj`、`action_out_proj` 共 66,592 个参数，两个 transformer 骨干和时间
MLP 仍冻结。它不增加权重键，但超出了原来约 0.05M 参数的方法预算，结果应单独报告。
此配置要求 CABO、PEFT 都关闭，并保存完整 checkpoint；部署电脑也要更新分支以识别新字段。

## 先在任务 4 的一个 episode 上检查拟合

默认只选 episode 0，正式 flow 训练 2,000 步，另有 750 步 priming 和 250 步 bridge，
只保存正式阶段最终 checkpoint。使用单卡、batch=8、FP32、预测 50/执行 8，关闭图像
增强；短实验默认关闭编译。所有 profile 使用同样的这些设置。

```bash
cd /root/lerobot060
git pull --ff-only origin piper

DRY_RUN=true bash examples/training/train_piper_fit.sh 4 projections 0

RUN_GROUP=piper-fit-task4-projections \
  bash examples/training/train_piper_fit.sh 4 projections 0
```

默认数据目录为 `/root/autodl-tmp/datasets/May-pick-and-place`，基础模型和 tokenizer
分别为 `/root/autodl-tmp/models/pi05_libero_base` 和
`/root/autodl-tmp/models/google/paligemma-3b-pt-224`。
非 AutoDL 机器可覆盖 `WORK_ROOT`、`DATASET_BASE`、`PRETRAINED_PATH`、`TOKENIZER_PATH`、
`OUTPUT_ROOT`、`LOG_ROOT`；和原 AutoDL 入口一致。各实验使用不同 `RUN_GROUP`。

该示例的输出目录是：

```text
/root/autodl-tmp/chkpt/2601-lerobot/piper/piper-fit-task4-projections/
  pi05-may-4-pick_the_red_cube_into_the_yellow_plate-no_cabo-seed0/
    training_diagnostics.jsonl
    checkpoints/002000/pretrained_model/
```

目录中的 `no_cabo` 是复用的底层 variant；`projections` 由运行组和保存的配置字段区分。
离线评价同一个训练 episode：

```bash
PIPER_FIT_RUN=/root/autodl-tmp/chkpt/2601-lerobot/piper/piper-fit-task4-projections
PIPER_TASK=4-pick_the_red_cube_into_the_yellow_plate

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python eval_piper_offline.py \
  --policy_path "${PIPER_FIT_RUN}/pi05-may-${PIPER_TASK}-no_cabo-seed0/checkpoints/002000/pretrained_model" \
  --dataset_root "/root/autodl-tmp/datasets/May-pick-and-place/${PIPER_TASK}" \
  --tokenizer_path /root/autodl-tmp/models/google/paligemma-3b-pt-224 \
  --episodes 0 --stride 1 --max_samples 0 --execution_steps 8 --plots 6 --seed 0 \
  --output_dir /root/autodl-tmp/eval/piper-fit-task4-projections-ep0
```

查看 `summary.json`、`metrics.csv` 和 `episode_*_frame_*.png`，重点比较第 1 步及前
8 步的逐关节 MAE（角度）和夹爪误差，并检查接近、闭爪、抬起等实际运动片段。
静止保持基线在大量静止帧上可能很好，不能仅凭全段平均值判断是否学会抓取。
输入包含同步的两路示范图像、当前 state 和 task，不能只给视频而漏掉 state。
也应使用现有 12,000 步权重先评价同一个 episode：替换上面的 `--policy_path`，
并使用另一个新的 `--output_dir`，其余采样及随机种子保持一致，保留改动前的对照。

再使用相同 episode、seed、步数运行其他 profile，选择不同 `RUN_GROUP`，进行相同评估。
一个 episode 的拟合模型仅用于诊断，不用于现场任务成功率评测：

- 连训练片段都拟合不好：继续查动作语义、相机对应、有效梯度/学习率和模型适应能力。
- 训练片段拟合较好，未见过的 episode 较差：再查泛化和训练覆盖。
- 离线误差较小，现场仍压桌：重点查相机视角/颜色、起始状态、动作跟踪和控制时序。

## 拟合有效后再训练全部数据

```bash
FIT_EPISODES=all FLOW_STEPS=12000 SAVE_STEPS='[6000,9000,12000]' \
  RUN_GROUP=piper-projections-full-01 \
  bash examples/training/train_piper_fit.sh 4 projections 0
```

将 `4` 换为 `all` 才会依次训练四个独立模型。正式阶段仍只保留 6,000、9,000、
12,000 步 checkpoint。若实验必须保持原来的 prompt-only 参数预算，使用
`prompt_lr` 或 `no_cabo` 进行相应对照，不使用 `projections`。

## 本次代码验证范围

在修改环境中通过了训练入口和诊断日志的 21 项测试、Python Ruff 检查及 shell 语法检查。
新增的真实 PI05 小模型梯度/冻结边界/权重回读测试在
`tests/policies/pi0_pi05/test_pi05_prompt.py`，但当前环境缺少 PyTorch，尚未运行。
没有用户的本地权重、数据和 GPU 训练环境，因此没有实测训练拟合或真机成功率。
具备项目测试依赖的机器可运行：

```bash
python -m pytest tests/policies/pi0_pi05/test_pi05_prompt.py -k projection_adaptation -q
```
