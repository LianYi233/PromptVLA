# 关键动作损失加权：保留旧 W0-Bocc 的样本与噪声

本实验先回答：**在完全相同的输入帧、动作片段和 ε/t 配对下，给夹爪事件附近的动作目标更大权重，是否提高成功率？** 然后比较前期、后期强调关键动作的训练课程。它不会重新采样输入图像，也不把高难度与关键性混为一谈。

之前 W0-Bocc/P1/D0 的 E1/T1 达到70.50%，E0/T0 达到61.50%。新 B0/B1 用相同旧计划复测，不能只用历史数字作为对照。FrameSkip U/R/S/C 用过新的有限样本池、发生次数编号及采样规则，不能拿本实验和它们作只改变一个因素的比较。

## 固定条件

- 原始 `pi05_libero_base`，仅训练2个 VLM prompt（4096参数），加载本机原始 P1 初始化；其余权重冻结。
- D0/W0，**原始** `orderbind-r1-occ/D0-W0.npz` 的192000个样本使用实例、顺序、任务槽位与 occurrence ID 全部保留。
- E0/T0 或 E1/T1：分别控制 flow matching 的动作噪声和插值时间的两个生成器，这里联合切换为 bank0/1；不是完整训练 seed 的两次独立重复。
- `Bocc`：ε/t 随 occurrence ID 移动，不改变分布、生成函数或 generator namespace。
- 两卡／模型，batch32／卡，全局64；3000更新，bf16，compile=false；仅保存完整003000模型。小型 prompt、秩探针和审计文件仍保存。
- 保留原 LR 配方：配置 warmup1000、decay30000，在3000更新时调度器等比缩放为有效 warmup100。**没有 FrameSkip 的额外500步数据 warmup**。
- 不改变连续50步 action chunk、归一化、padding、任务顺序、图像预处理；不使用 action prompt、CABO、priming 或 bridge。

## 事件定义与损失

直接读取已有 `/data1/wyn/prompt-work/frameskip-r1/scores.npz` 的 `event_neighborhood`：夹爪开合切换前10帧到后2帧、debounce=2。窗口在 episode 边界截断。此处不用视觉评分的 top20%选择，也不需要重新做 GPU 离线评分。

这是夹爪事件的代理标签，无法覆盖所有推、拉、插入或移动中的关键动作，也不是难度标签。先检查原来的事件预览（建议每个 task 至少2段），确认开合方向、窗口与真实操作对应。没有事件的 task 保持普通动作监督，不强行挑选固定百分比。

每个输入帧原来对应50步动作目标。例如输入帧100的目标为100…149；事件窗口110…122只会提升这些目标的权重，输入图像仍是帧100。padding目标永远不参与损失。所有7个真实动作坐标采用同一个时间步权重，模型的补齐维度不参与。

| 模式 | 关键动作权重 | 其他有效动作权重 | 回答的问题 |
|---|---:|---:|---|
| B | 1 | 1 | 原始基线能否复现？原版 loss reduction 原样执行 |
| K2 | 2 | 1 | 固定强调关键动作是否有效？ |
| R2 | 随机窗口2 | 1 | 收益来自正确动作位置，还是一般的非均匀加权？ |
| Kearly | 前高后低，名义2.5→1.5，实际校准 | 1 | 训练前期强调关键动作是否更好？ |
| Klate | 前低后高，名义1.5→2.5，实际校准 | 1 | 训练后期强调关键动作是否更好？ |

R2 将每个原始动作片段的事件 mask 在其有效时间轴上**循环平移**。每次样本使用的平移量由私有 RNG2026 预生成并保存，P/D/E/T 四个因素均不受影响。每个样本、batch、task 的被加权目标数量与 K2 完全相同，因此分母也相同；只改变被强调的动作位置。边界处窗口可能拆开，也可能与真实事件重叠，重叠比例会记录。它检验的是相同观测条件下的目标位置，不是随机挑选另一批输入帧。

令全局 batch 中有效动作时间步数为 V，选中目标数为 K，权重为 w，坐标数为 d=7：

`loss = sum(valid * weight * squared_error) / (d * sum(valid * weight))`

DDP 用各卡真实加权分母进行修正，使结果对应整个64样本 batch 的加权平均，而非错误地平均两个不等分母的局部均值。

课程曲线采用 `w=1+c*g(update)`，early的g从1.5线性降到0.5，late反向；各自求解c，使3000步的 `sum(w*K/(V+(w-1)*K))` 与K2相同。实际首尾权重在 `summary.json`，每步值在 `schedule.csv`。这匹配**归一化后关键项的总系数预算**，不声称匹配梯度范数、LR加权影响或优化轨迹。此课程改变重要性权重，不是“简单→困难”的排序课程。

## 一次 CPU 准备

在安装了本仓库的 lerobot 环境、仓库根目录执行：

```bash
cd /home/wyn/VLAA/PromptVLA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m lerobot.scripts.keyframe_experiment prepare \
  --scores /data1/wyn/prompt-work/frameskip-r1/scores.npz \
  --old-plan /data1/wyn/prompt-work/orderbind-r1-occ/D0-W0.npz \
  --output /data1/wyn/prompt-work/keyframe-r1
```

输出 `archive.npz`、原样复制的 `D0-W0.npz`、`summary.json`、`schedule.csv`。两个训练队列共用**已经准备完成**的只读文件，不要在两个终端同时运行 prepare。

如果旧计划不在默认路径，用 `--old-plan` 指向旧 W0-Bocc 运行的 `pdet/manifest-rank0.json` 中 `factors.order_plan` 所记文件。旧 slot 队列的 W0 文件也可用，前提是其完整 manifest/array hash 与旧 occ 文件相同。**不要用 FrameSkip U.npz 替代，也不要重新编号使用实例**；旧计划丢失时应先恢复或通过历史审计确认重建完全一致。

准备时及训练启动前核对 dataset metadata、actions、stats 的哈希和 source/video 的文件大小+mtime清单。后两者不是完整权重/视频内容哈希。

## 四卡训练

清除之前其他实验留下的覆盖变量；以下命令默认 W&B online，project=`prompt-learning`，不上传 artifact。额外只接受 `--wandb.*=...`，避免无意改动实验配方。

终端A（GPU0、1）：

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES -u CELLS \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29571 \
bash examples/training/train_pi05_keyframe_curriculum.sh a baseline
```

终端B（GPU2、3）：

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE -u PDET_CASES -u CELLS \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29572 \
bash examples/training/train_pi05_keyframe_curriculum.sh b baseline
```

先测试 B0/B1。如果复现严重偏离旧结果，先查数据、初始化、归一化和训练诊断，再追加四个加权模型。确认后把两条命令最后的 `baseline` 分别改为 `weight`；随后如需比较课程，再改为 `curriculum`。每个阶段仅运行新模型，不自动重跑已完成阶段。

| 阶段／轮次 | 终端A：GPU0、1 | 终端B：GPU2、3 |
|---|---|---|
| baseline | B，E0/T0 | B，E1/T1 |
| weight第1轮 | K2，E1/T1 | K2，E0/T0 |
| weight第2轮 | R2，E0/T0 | R2，E1/T1 |
| curriculum第1轮 | Kearly，E1/T1 | Kearly，E0/T0 |
| curriculum第2轮 | Klate，E0/T0 | Klate，E1/T1 |

命名示例：`pi05-key-r1-K2-W0-Bocc-base-vlm2-P1-D0-E1-T1`。模型保存到 `/data1/wyn/chkpt/2601-lerobot/PromptVLA/<name>/checkpoints/003000/`，完整训练日志在 `/data1/wyn/chkpt/2601-lerobot/logs/<name>.log`。

`DRY_RUN=true` 可做 CPU 启动预检。输出目录已存在时拒绝覆盖，当前版本不支持这些实验的断点续训。中断后用 `CELLS="R2:0"` 选择原队列／阶段中尚未开始的模型；失败的已有目录先手动改名保留，再从头重跑相同 cell。不要改初始化或训练随机因素来绕过冲突。

## 审计与 LIBERO 测试

两组B训练完成后：

```bash
python -m lerobot.scripts.keyframe_experiment audit --stage baseline \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-key-r1-B-W0-Bocc-base-vlm2-P1-D0-E*-T*

GPU_ID=0 CELLS="B:0" bash run_eval_libero_keyframe_curriculum.sh baseline all
# 另一个终端并行：
GPU_ID=2 CELLS="B:1" bash run_eval_libero_keyframe_curriculum.sh baseline all
```

六组 B/K2/R2 完成后，audit的 `--stage` 改为 `weight`，路径选 `pi05-key-r1-*-W0-Bocc-base-vlm2-P1-D0-E*-T*`。十组都完成后使用 `--stage curriculum`；不要把其他实验／重复轮次混入一个比较。

四个加权模型可单卡并行评估，四个终端分别执行：

```bash
GPU_ID=0 CELLS="K2:0" bash run_eval_libero_keyframe_curriculum.sh weight all
GPU_ID=1 CELLS="K2:1" bash run_eval_libero_keyframe_curriculum.sh weight all
GPU_ID=2 CELLS="R2:0" bash run_eval_libero_keyframe_curriculum.sh weight all
GPU_ID=3 CELLS="R2:1" bash run_eval_libero_keyframe_curriculum.sh weight all
```

课程评估对应 `curriculum` 阶段、`CELLS="Kearly:0"` 等。所有评估默认 checkpoint3000、seed1000、每任务10次、四suite共400次，单行进度条与最终成功率；视频路径和完整输出留在日志。CELLS不指定时该阶段依次评估。

审计核对所有 step/rank 的真实样本和 ε/t hash、相同初始化/source/recipe、相同事件缓存、R2掩码、加权分母及课程预算。它不证明 GPU 完全数值确定性，也不证明当前代码与旧历史版本等价。运行结束打印的 `partial_inputs_passed` 仅表示该队列所选子集通过；两个队列都完成后再运行上面的完整审计。

## 记录与判断

每个运行保存：

- `pdet/`：完整样本、occurrence ID、ε/t hash、参数更新诊断和代码来源。
- `keyframe_loss/archive.npz` 与 `manifest-rank*.json`：独立保存事件、随机mask、课程与原计划元数据。
- `keyframe_loss/inputs-rank*.jsonl`：每步权重、真实关键mask的hash、valid/key计数和分母。
- `keyframe_loss/training-rank*.jsonl`：每10步各卡的关键／非关键平方误差总和及坐标数、未加权和加权训练MSE；汇总应先加总分子分母，不能直接平均不同分母的局部MSE。
- `prompt_rank/`：原有固定16个训练观测、私有noise/projections，0/50/100/250…3000步的有效秩及固定探针**未加权**关键／非关键MSE。复用同一次前向计算，不新增训练随机数抽样。探针是训练观测诊断，不是验证集或rollout指标；没有关键目标时该项MSE省略并记录计数0。

先比较同bank的 K2−B 和 K2−R2；再比较 Kearly/Klate−K2。按四个suite、task以及整体报告，保留每episode结果以作配对分析。只有K2胜过B与R2且跨bank一致，才支持关键位置有价值；只有课程胜过固定K2，才支持时间安排的额外收益。两个bank是小规模筛选，不足以宣称普遍结论；最终方案还需新P初始化、更多bank/独立训练重复与固定评估协议复核。
