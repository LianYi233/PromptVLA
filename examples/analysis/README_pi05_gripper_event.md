# LIBERO-long：单条示范的夹爪闭合采样与动作时间编码

本实验检查两个独立问题：增加夹爪闭合附近观测的训练曝光是否有帮助；在动作
Transformer 的输入表示中加入相对时间编码是否有额外帮助。

这里的“一个样本”按**一条完整示范 episode**实现。在该 episode 内，可以选择
许多观测帧作为训练起点。若只固定一个观测帧和它之后的一段动作，则没有可改变的
起点采样分布，无法检验本次假设。

这是一项新实验，基于 `feat/keyframe-loss-curriculum` 的代码；不会自动调用旧的
难度评分、H2E/E2H 课程、关键帧损失加权、CABO、priming 或 bridge。

## 1. 具体做了什么

### 数据范围与动作连续性

- 默认选择官方 `libero_10`（LIBERO-long）任务 8：
  `put both moka pots on the stove`（把两个摩卡壶放到炉子上），
  完整任务名为 `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`。
- 在该任务中按 episode index 从小到大选择第一条存在可用闭合事件、且同时包含
  普通起点与事件邻域起点的示范；可用 `--episode-index` 显式指定。
- 任务根据 `meta/tasks.parquet` 或 `meta/tasks.jsonl` 的精确语言匹配确定。
  混合数据集的 `task_index` 与官方 suite 的 task ID 分别记录，不假定两者相同。
- 每次抽中原始帧 `t`，模型接收该帧已有的相机观测、状态和语言，并监督原始连续
  动作 `a[t:t+50]`。**不抽取一个跳帧的视频，也不在动作 chunk 内重排或稀疏抽样。**
- 排除 episode 尾部不足 50 步的起点，防止 padding 或跨 episode 混入对照。
  尾部动作仍可以出现在前面合法起点的连续标签中。

更频繁抽取某些起点，本身不会破坏 chunk 内的时间顺序。原动作 Transformer
已经有 token 位置表示；新增时间结构是独立的归纳偏置实验，不是保证有效的修复。

### 闭合事件与采样概率

在**原始、未归一化的 action 最后一维**上检测稳定的 open→closed 指令转变。
默认 `closed-direction=increasing`，与原始 Panda 控制约定的 `-1=open,+1=close`
一致。转换过方向的数据必须指定 `--closed-direction decreasing`。

- 默认至少连续 2 帧确认状态；初始已闭合、持续闭合和短时抖动不重复计数。
- 二值 `{-1,+1}` 自动使用阈值 0，`{0,1}` 自动使用阈值 0.5。
- 连续值或其他编码要求显式 `--gripper-threshold`，不能猜测数据的语义。
- 检测的是“发出闭合指令”，并不能证明真实接触或抓取成功。
- 无有效闭合事件或整个候选池都属于事件窗口时会拒绝准备，避免生成实际没有
  采样干预的 event 对照。

默认窗口为事件前后各 5 帧，窗口内权重 3，其他合法起点权重 1，然后归一化为概率。
重叠窗口取并集，不累计加权。所有合法普通起点仍有非零抽样机会。

例如 100 个候选起点中 10 个在闭合邻域，uniform 抽到这些起点的概率是 10%，
event 则是 `3*10/(3*10+90)=25%`。权重 3 指同一 event 分布内事件起点相对
普通起点的概率比，不代表事件总概率一定变为均匀采样时的 3 倍。

prepare 会分别报告：

- `event_anchor_probability`：起点落在闭合邻域的理论概率；
- `closure_in_chunk_probability`：后续完整 50 步标签中含有闭合事件的理论概率；
- `realized_*`：有限训练计划中实际抽取后的对应比例；
- `draws`、`unique_sampled_anchors`：总曝光和不同起点覆盖数。

这两个覆盖率不能混为一谈：事件前较早的起点，也可能在其后续 50 步标签中包含闭合。
本实验只改变起点分布，不另做 loss 加权，也不使用逆概率权重抵消采样干预。

### 时间编码放在模型输入表示中

每个动作 token 使用相对物理偏移 `delta_t[j] = j / dataset_fps`，经 32 维正弦/
余弦特征和可训练线性投影后，加到 noisy-action embedding 上。数据 FPS 从
`meta/info.json` 读取，并写入 checkpoint；不默默假设所有数据都是 20 FPS。

标准 `gemma_300m` 的新增参数为 `1024*32=32768`，无 bias。投影用全零 Parameter
构造，不消耗随机数，因此开关两组在训练开始时有相同的共有权重、prompt、随机流
和模型预测。训练和推理都经过同一个 `embed_suffix` 路径。

动作 label 仍是原来的 7 维，模型内部仍按原配置补齐维度。这里没有把时间作为一个
新的机器人控制量，也不输入示范绝对时刻、episode 总长度或“距未来闭合还有多久”。
物理时间偏移与 flow-matching 的随机噪声时间是两种独立变量。

旧 checkpoint 默认关闭该结构。加载原始 source 来开启实验时，只允许新增零权重；
已保存为开启状态的 checkpoint 若缺少时间权重，会明确报错。

## 2. 四组实验与固定设置

| Case | 起点采样 | 相对时间编码 | 使用的计划 |
| --- | --- | --- | --- |
| `uniform` | 均匀、有放回 | 关闭 | `uniform.npz` |
| `event` | 闭合邻域加权、有放回 | 关闭 | `event.npz` |
| `uniform_time` | 均匀、有放回 | 开启 | 同一个 `uniform.npz` |
| `event_time` | 闭合邻域加权、有放回 | 开启 | 同一个 `event.npz` |

默认每组：`pi05_libero_base`，VLM prompt 2 tokens + **完整 action expert**，冻结
视觉/VLM 主干，原始 P1 初始化，global seed 0，D0/E1/T1，3000 optimizer updates，
2 GPU × 每卡 batch32，bf16、gradient checkpointing、compile=false，仅保存最终
`003000` 完整模型。学习率沿用当前 base wrapper：`2.5e-5`、原有 warmup/cosine
缩放规则。time 组额外训练上述 32,768 参数。W&B 默认 online、project=prompt-learning。

这是完整动作网络继续适配，不是 prompt-only 参数预算实验。默认四组顺序运行，
每组使用 GPU0/1，不会自动占用另外两张 GPU。

uniform/event 改变训练样本池；它们共享更新预算和每个曝光位置的 flow 噪声/时间。
同一采样条件的 time off/on 进一步共享完全相同的帧顺序。每次更新记录实际输入，
训练结束核对两卡采样计划、曝光预算、prompt、模型来源和各组噪声/时间哈希。
`REPEAT_ID` 仅改变实验名称；改变它不等于新增随机种子试验。

## 3. 准备数据并启动

先在已配置的服务器环境中切换到新分支，安装当前 checkout：

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
git -c core.sshCommand="ssh -p 443" fetch \
  git@ssh.github.com:LianYi233/PromptVLA.git feat/gripper-event-time-long
git switch -c feat/gripper-event-time-long FETCH_HEAD
python -m pip install --no-deps -e .
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

如果本地已经有该分支，把上面的 `git switch -c ...` 替换为以下两条命令，
切换后快进到刚刚 fetch 的版本：

```bash
git switch feat/gripper-event-time-long
git merge --ff-only FETCH_HEAD
```

默认路径沿用当前仓库：

| 资源 | 路径 |
| --- | --- |
| 数据 | `/home/wyn/datasets/libero` |
| 模型 | `/home/wyn/models/lerobot/pi05_libero_base` |
| tokenizer | `/home/wyn/models/paligemma-3b-pt-224` |
| prompt | `/data1/wyn/prompt-work/pi05_libero_base-vlm2-init-seed1.safetensors` |
| 准备目录 | `/data1/wyn/prompt-work/gripper-event-long-task8-r1` |
| checkpoint 根目录 | `/data1/wyn/chkpt/2601-lerobot/PromptVLA` |
| 训练日志根目录 | `/data1/wyn/chkpt/2601-lerobot/logs` |

P1 也兼容旧命名 `pi05-base-vlm2-init-seed1.safetensors`；若两个名字存在且内容不同，
会拒绝自动选择。`--prompt /path/to/P1.safetensors` 可指定已有、带 provenance 的原始
初始化，不依赖旧 source-pair 实验或 generic source 模型。

### 第一步：固定一条轨迹和采样计划

```bash
export GRIPPER_EVENT_DIR=/data1/wyn/prompt-work/gripper-event-long-task8-r1
python -m lerobot.scripts.pi05_gripper_event prepare \
  --dataset-root /home/wyn/datasets/libero \
  --output "$GRIPPER_EVENT_DIR" \
  --task-id 8 \
  --steps 3000 --batch-size 32 --world-size 2 \
  --window-before 5 --window-after 5 --event-weight 3
```

输出 `manifest.json`、`uniform.npz`、`event.npz`、`anchors.csv`。查看终端中的
`episode_index`、`closure_frames` 和采样比例。想固定其他轨迹可增加
`--episode-index N`；`N` 是完整数据集的 episode index，并非 task 内排序号。
同一准备目录只允许完全一致的重复准备，修改参数需使用新目录。
若此前已按旧默认准备了 task 0，使用上面的 task8 目录重新准备；显式导出
`GRIPPER_EVENT_DIR` 后，训练、评测和汇总都会使用新的 task8 清单。

所有帧索引保留完整数据集的 global index；训练只使用计划引用的那一条 episode。
不要额外传 `dataset.episodes` 或重编号，否则会破坏计划和 dataset 的索引对应。
归一化统计沿用完整训练数据集统计，四组一致。

### 第二步：检查四组命令，然后训练

```bash
DRY_RUN=true bash examples/training/train_pi05_gripper_event.sh all
GPU_IDS=0,1 bash examples/training/train_pi05_gripper_event.sh all
```

可先只跑最直接的采样对照：

```bash
bash examples/training/train_pi05_gripper_event.sh uniform
bash examples/training/train_pi05_gripper_event.sh event
```

再单独补 `uniform_time`、`event_time`。已存在的训练输出会报错，不覆盖、不默默跳过。
若队列中途停止，完成的 case 保留，继续时显式选择未启动的 case。默认不实现失败
训练的自动断点恢复；换新 repeat 标签会从相同初始条件重新训练该 case。

`--wandb-mode offline` 或 `--wandb-mode disabled` 可用于无法访问 W&B 的环境。
`--gpus 2,3 --port 29590` 可以把**整组双卡实验**放到另外两张 GPU。

### W&B 初始化失败后重跑

若日志在 `WandBLogger` → `wandb.init` 处显示 `403 Forbidden`，说明在线创建 W&B
run 失败。该阶段早于数据集、模型构造和 optimizer 更新；其他 rank 后续出现的
NCCL 超时可能是主进程失败后未继续同步的结果。训练入口会协调传播 W&B 初始化
异常，让所有 rank 退出并保留原始错误，而不会自动更改日志模式。

需要保留可在以后上传的 W&B 指标时，显式使用 `--wandb-mode offline`。本实验
wrapper 显式传入 W&B mode，因此不要只设置 `WANDB_MODE` 环境变量。
初始化失败仍可能留下输出目录和 `runs/<run_name>.json` 注册记录；改日志模式后
应使用新的 `--repeat` 编号，保留旧记录，无需重新 prepare：

```bash
DRY_RUN=false GPU_IDS=0,1 \
bash examples/training/train_pi05_gripper_event.sh all \
  --directory /data1/wyn/prompt-work/gripper-event-long-task8-r1 \
  --repeat 2 --wandb-mode offline
```

若希望完全跳过 W&B SDK 初始化，将最后的 `offline` 改为 `disabled`。这时仍保留
终端/文件训练日志、`training_diagnostics.jsonl`、PDET 审计和 checkpoint，但不生成
可供之后同步的 W&B run。`--repeat 2` 只改变 run 名称；继续使用原 task8 episode、
采样计划、随机种子和训练预算。若编号 2 已被使用，选另一个未使用的正整数。

训练完成后，评测和汇总需要使用同一编号：

```bash
GPU_ID=0 bash run_eval_libero_gripper_event.sh all \
  --directory /data1/wyn/prompt-work/gripper-event-long-task8-r1 --repeat 2
python -m lerobot.scripts.pi05_gripper_event_experiment summarize all \
  --directory /data1/wyn/prompt-work/gripper-event-long-task8-r1 --repeat 2
```

### P1 初始化（仅在文件不存在时）

如 P1 文件尚不存在，先用已有初始化导出入口创建，零次训练更新：

```bash
TRAIN_ACTION_EXPERT=false FREEZE_VLM_PROMPT=false \
VLM_PROMPT_TOKENS=2 GPU_IDS=0 SAVE_ONLY_FINAL=true \
RUN_NAME=gripper-long-export-P1 \
PROMPT_EXPORT_PATH=/data1/wyn/prompt-work/pi05_libero_base-vlm2-init-seed1.safetensors \
bash examples/training/train_pi05_vlm_only.sh base 1 --policy.compile_model=false
```

该初始化会正常构造原模型，需使用模型所在环境；不是在 CPU 准备采样时伪造随机张量。

## 4. 只在 LIBERO-long 评测

```bash
GPU_ID=0 bash run_eval_libero_gripper_event.sh all
```

默认每个 task 10 episodes、eval seed1000，四组使用相同初始条件；每次只在一个
GPU 上依次评测。保持 checkpoint 中的时间编码设置，执行步数沿用现有评测的
`n_action_steps=10`。可用 `EPISODES_PER_TASK=20` 增加评测次数。
评测仅支持标准 `libero_10`；若继承了 PLUS/PRO 环境覆盖会要求清除后重试。

结果目录包含任务、episode、准备清单短哈希、repeat、评测 seed 和回合数，防止
不同小实验互相覆盖。默认位于：

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-event/
  task8-ep<episode>-<manifest_hash>/r1-seed1000-n10/
    summary-all.json
    summary-all.csv
    libero060-all-<run_name>-003000-libero_10-resume/eval_info.json
```

脚本会打印完整路径。训练日志在 `$LOG_ROOT/<run_name>.log`；checkpoint 在
`$OUTPUT_ROOT/<run_name>/checkpoints/003000/pretrained_model/`。评测沿用原入口的
任务级进度保存，重跑相同命令可继续未完成的任务。

若此前分开评测四组，可统一汇总：

```bash
python -m lerobot.scripts.pi05_gripper_event_experiment summarize all
```

汇总器要求完整十任务、每任务达到约定回合数；不会把中断后的部分结果当成总体成绩。
它同时报告训练示范所属 task 和整个 LIBERO-long 的成功率，并计算：

- `event - uniform`：闭合邻域采样的贡献；
- `uniform_time - uniform`：均匀采样下的时间结构贡献；
- `event_time - event`：事件采样下的时间结构贡献；
- `(event_time - event) - (uniform_time - uniform)`：二者交互量。

`GRIPPER_EVENT_DIR`、`OUTPUT_ROOT` / `CKPT_ROOT`、`LOG_ROOT`、`TOKENIZER_PATH`、
`BASE_OUTPUT` 等参数支持自定义；`--help` 列出对应 CLI 参数。必须使用与 prepare
一致的数据、steps、batch 和 GPU 数，不能单靠 `FLOW_STEPS` 等旧环境变量改预算。

## 5. 能支持怎样的结论

该实验适合先检验实现和单示范适配行为。优先看训练任务在多个标准初始状态下的
闭环成功率，并同时观察其他九任务是否退化。单条训练轨迹的 loss 更低，不能单独
证明闭合采样改善了实际执行。

`pi05_libero_base` 与完整训练集统计都来自既有 LIBERO 配置，因此这里是**对已有
LIBERO source 的单示范继续适配**，不能称为模型首次通过一条示范学会未见任务。
一条 episode、一个种子也不能证明对整个 LIBERO-long 普遍有效。默认每任务10回合
的成功率以10个百分点跳变，小差值需要更多初始状态/独立示范和种子验证。

固定 FPS、固定连续50步时，相对物理时间由序列位置确定，没有引入新的观测信息。
如 event 改善而 event_time 没有额外改善，合理解释是采样有效而附加时间结构冗余；
不能因此否定采样假设。若只有训练loss改善，可能只是对单轨迹或事件窗口过拟合。

## 6. 开发验证与尚未运行的内容

新增测试覆盖真实 tiny-PiGemma CPU forward/backward/推理、零初始化与RNG一致性、
完整 checkpoint 保存加载、旧模型兼容、合成 parquet 事件检测/任务筛选/动作连续性、
PDET计划与实际shell dry-run、完整十任务结果校验和报告隔离。

本分支交付时尚未在用户的 A100 上执行真实 PI0.5 训练或 MuJoCo/LIBERO 闭环评测，
仓库不包含虚构的成功率或假造的真实数据 manifest。运行以上命令后才能得到实验结论。

实现依据：[LIBERO 官方任务列表](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py)、
[LeRobot LIBERO 文档](https://huggingface.co/docs/lerobot/libero)、
[robosuite gripper 约定](https://robosuite.ai/docs/source/robosuite.models.grippers.html)、
[OpenPI 原始位置与 flow-time 路径](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)。
