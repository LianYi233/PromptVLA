# VLA curriculum motivation diagnostics

基于 `feat/keyframe-loss-curriculum` 的 `c3dfb61` 接口开发。新功能全部位于
`examples/analysis/curriculum_study/`；**不修改旧的 trainer、模型、采样器、A–H 启动脚本或审计规则**。
旧命令的默认行为保持不变。本入口也不会接着旧模型继续训练。

## 功能与证据边界

| 入口 | 功能 | 不能据此声称什么 |
| --- | --- | --- |
| `legacy` | 从已有 `prompt_rank` 日志绘制低分/高分训练探针的二维 loss 轨迹 | 不是独立验证集，不是控制风险，不是参数空间的信号/噪声方向 |
| `prepare/run/audit/plot` | 3 种顺序 × 2 种 LR；episode 留出；固定 probe；全步输入与实际 LR 审计 | 恒定 LR 不会消除 Adam 历史、遗忘等所有顺序机制 |
| `eval` | 对新实验指定的完整 checkpoint 运行原标准 LIBERO 评测 | 不能仅用离线 loss 代替闭环成功率 |
| `perturb` | 同一前缀重放后的状态，配对正常/单步扰动/零扰动 continuation | 只测指定 baseline、状态、扰动尺度下的控制敏感性 |
| `semantic` | 可选的冻结外部 PaliGemma 多选题探针 | 不是 π0.5 的真实语义能力，也不是自动生成的 ground truth |
| `risk-plot` | 按 `state_uid` 一对一连接测量结果并画散点图 | 图形本身不能证明 H2E 必然有效 |

所有图使用实测输入；没有 toy loss landscape、伪造轨迹、插值补点或“已知最优点”。
导出 PNG、SVG、CSV/JSON。原始训练输出只读；分析目标目录已存在时拒绝覆盖。

## 环境

在原先能训练/评测 PI05 的环境与仓库根目录下运行：

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

CPU 分析需要 NumPy、Matplotlib；读取历史缓存/探针测试需要 PyTorch；测试需要 pytest。
训练沿用已安装的 LeRobot、Accelerate、Transformers 等依赖；不要为了这些入口升级原环境。
没有附带模型、LIBERO 数据或实际实验结果。

## 1. 先画已有 A/B/C 的训练轨迹：不用重训

```bash
python -m examples.analysis.curriculum_study legacy \
  --jobs /data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-steps6000/jobs.json \
  --cells A B C \
  --scores /data1/wyn/prompt-work/frameskip-r1/scores.npz \
  --baseline-plan /data1/wyn/prompt-work/source-pair-r1/D0-W0.npz \
  --weighting micro \
  --output /data1/wyn/prompt-work/analysis/legacy-e30-r2
```

D/E/F 使用同一命令，改 `--cells D E F` 和一个新的输出目录。**不要把不同曝光比例的组
混成“纯顺序对照”**；程序会检查 occurrence pool、任务时隙、训练预算与 probe 缓存一致性。
也支持 `--runs H2E=/完整run路径 E2H=/完整run路径 RND=/完整run路径`。

历史默认只有 16 个训练 probe，通常不满足两组覆盖相同任务的条件，所以上例明确选择
`micro`（有效坐标加权，探索性分析）。默认 `task` 要求两组覆盖同一批任务，并先按样本、
episode、task 分层平均。新实验的 probe 会按 task/group 均衡构造。

默认要求 `fixed_probe.pt`，并比较实际缓存的观测与噪声。`--allow-unverified` 只用于缺失缓存时
查看探索性图像，会在 provenance 中标明未完成配对验证；不应作为正式机制证据。

## 2. 新建顺序 × LR 的六组实验

```bash
python -m examples.analysis.curriculum_study prepare \
  --output /data1/wyn/prompt-work/curriculum-study-v1 \
  --steps 6000 --easy-percent 30 \
  --holdout-per-task 2 --probe-per-group 2 \
  --warmup-steps 200 --probe-frequency 250 \
  --checkpoints 1000,3000,6000
```

默认复用原 `source-pair-r1`、`frameskip-r1/scores.npz`、
`/home/wyn/datasets/libero`、`/home/wyn/models/paligemma-3b-pt-224`。
可通过 `--source-pair/--scores/--dataset-root/--tokenizer` 覆盖。

### 严格固定什么

- 先在每个任务中随机留出完整 episode，再从剩余的原候选样本构造训练 pool。
- H2E/E2H/RND 使用完全相同的帧使用次数、occurrence ID、任务时隙和每次使用绑定的 ε/t。
- 前 200 次更新使用**完全相同的随机 warmup 前缀**；随后才对剩余 occurrence 做 task 内排序。
- 两种 LR 都使用相同的实际 warmup。cosine 为 `2.5e-5 → 1e-5`；constant 在 warmup 后保持 `2.5e-5`。
  显式令 `scheduler_decay_steps=6000`，避免名义 1000 warmup 被自动缩放却未记录。
- 两个 VLM prompt + 完整 action expert 都训练；VLM/视觉主干冻结。没有 CABO、priming、bridge、loss weighting。
- 全部从同一个 source 和同一个 P1 重新初始化，不继承前一组训练结果。

这里的留出指**本轮微调 optimizer updates 不使用这些 episode**。源模型可能在预训练中见过它们，
评分及原归一化统计也不是重新在 train-only 数据上拟合；因此不声称“从未见过的严格泛化集”。
所有新组使用同样的既有处理器统计。由于新 pool 留出了 episode，不能把新组与历史 A–H 的差异
归因于 LR 或顺序；应在新六组内部比较。

默认 probe 标签是 `low_score/high_score`，不自动叫 `routine/critical`。
已有独立事件标注时，可传 `--labels labels.csv`，格式为 `index,group`，group 为 `routine/critical`。
标注必须覆盖每个任务预先随机留出 episode 中足够的候选帧；不足时明确报错，不重新挑“好看”的 episode。

`--seed` 才改变重复实验随机性；单纯换输出目录不是独立 seed 实验。

### 先做 CPU 预检，再启动

```bash
python -m examples.analysis.curriculum_study run \
  --study /data1/wyn/prompt-work/curriculum-study-v1 \
  --all --gpus 0,1 --dry-run

# 六组依次运行，每组使用两张卡；任何一组失败即停止，不覆盖、不隐式续训。
python -m examples.analysis.curriculum_study run \
  --study /data1/wyn/prompt-work/curriculum-study-v1 \
  --all --gpus 0,1
```

也可在两个终端分配互不重叠的两卡：

```bash
python -m examples.analysis.curriculum_study run \
  --study /data1/wyn/prompt-work/curriculum-study-v1 \
  --order H2E --lr cosine --gpus 0,1 --port 29620

python -m examples.analysis.curriculum_study run \
  --study /data1/wyn/prompt-work/curriculum-study-v1 \
  --order H2E --lr constant --gpus 2,3 --port 29621
```

训练中不要修改代码/输入。已存在的 run 无论是否完整都不会被自动删除或重训。
新入口目前不支持中途训练 resume；中间完整 checkpoint 保留给评测与人工恢复用途。
完整 checkpoint 包含优化器，磁盘开销较大；只需要在线 loss 曲线时可在准备阶段设 `--checkpoints 6000`。

### 输出、审计与画图

每组目录下有 `study_spec.json`，以及：

```text
curriculum_probe/
  manifest.json          # 样本身份、分组、固定 noise/time 与处理后输入 hash
  fixed_probe.pt         # CPU 缓存；只加载可信的本项目产物
  step_000000.json        # 初始探针，之后按频率/额外节点保存
  step_000250.json
  ...
  lr.jsonl               # 每次更新的 used_lr 与 next_lr，二者明确区分
```

probe 不进行 backward、不更换 Parameter、不写 optimizer，也保存/恢复 RNG 与各子模块 train/eval 标志。
原 `update_policy` 仍完整负责每一次训练更新。任何非有限梯度导致的更新跳过都会停止实验，避免假装预算匹配。

```bash
python -m examples.analysis.curriculum_study audit \
  --study /data1/wyn/prompt-work/curriculum-study-v1
```

它检查六组的完整 PDET realized trace、实际 LR、完整 checkpoint、处理器和固定 probe，不仅比较 seed。

```bash
S=/data1/wyn/prompt-work/curriculum-study-v1
python -m examples.analysis.curriculum_study plot \
  --runs \
    H2E="$S/runs/pi05-study-H2E-cosine-e30-seed0-steps6000" \
    E2H="$S/runs/pi05-study-E2H-cosine-e30-seed0-steps6000" \
    RND="$S/runs/pi05-study-RND-cosine-e30-seed0-steps6000" \
  --output "$S/figures/cosine"
```

另一栏将所有 `cosine` 改为 `constant`，输出到独立目录。坐标是固定两类 probe 的实测 flow MSE，
不是参数 PCA，也不是全局最优点。图中圆圈为起点，X 为终点，箭头表示训练推进。

### 评测保存的任意里程碑

```bash
python -m examples.analysis.curriculum_study eval \
  --study /data1/wyn/prompt-work/curriculum-study-v1 \
  --order H2E --lr cosine --step 3000 --gpu 0 --suite all
```

沿用原标准 LIBERO 入口及其相机/步数设置，加载完整学习后权重和保存的处理器；不替换回原 base 的 action expert。

## 3. LIBERO 单步配对扰动实验

先做最小 smoke test，checkpoint 使用一个有基本完成能力的**随机顺序 baseline**，而非仅挑最好的 H2E：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.analysis.curriculum_study.perturb \
  --checkpoint /你的随机顺序baseline/checkpoints/006000/pretrained_model \
  --tasks libero_spatial:0 --episodes 1 --trials 1 --anchor-steps 20 \
  --delta 0.05 --dims 0,1,2 \
  --scores /data1/wyn/prompt-work/frameskip-r1/scores.npz \
  --output /data1/wyn/prompt-work/control-sensitivity-smoke
```

确认实际环境跑通后，新建一个输出目录，使用 `--tasks libero_spatial:0,libero_goal:0 --episodes 5
--trials 4 --anchor-steps 20,60,100,140`。默认与原标准 LIBERO 一样使用 360 图像和 10 步执行队列；
这些设置记录在 protocol 中，可显式调整，但所有对照必须一致。

### 实验实现

1. 从官方固定初始状态采集 baseline rollout，在预先指定时刻选状态，不按 importance 或成功与否挑状态。
2. 每条分支重新 reset 并重放相同的历史动作前缀，检查 simulator state 和完整观测 hash。
3. 只允许 action-chunk 边界；清空 policy/processor 缓存，固定配对推理 seed。
4. 每个 trial 运行 clean、zero 和 perturbed 三个 continuation。zero 必须与 clean 的完整状态/动作轨迹
   及成功结果匹配，否则停止，不输出该风险结果。
5. 扰动仅作用于一次动作的指定连续坐标，默认前三维；夹爪不变。默认所有分支同样将 baseline 命令
   限制到 [-1,1]，再用拒绝采样生成边界内的**精确范数**扰动，不通过裁剪悄悄缩小 δ。
6. 绕开原 `LiberoEnv.step` 的终止自动 reset，直接使用底层仿真 step，防止终止后误继续新 episode。

δ 的单位是**归一化控制命令**，不是米/角度；混合平移与旋转维度时不能赋予统一物理长度解释。
默认只扰动平移维度。此入口仅支持标准 LIBERO，不接真机，也不自动切换 PRO/PLUS。

记录所有可达 anchor；提前终止造成的不可达状态会写入 `coverage.json`，不会被静默换成成功样本。
`risk_summary.json` 中保留负风险和 baseline 全失败的地板效应状态。少量 trials 只用于探索；
同一 episode 的多个状态不是独立样本。

输出以 pair 为单位原子保存。相同命令可以复用已完成 pair，但会重新核对协议、模型/处理器、场景、
代码、参考 rollout 身份和零扰动条件；不承诺跨硬件逐位确定性，不一致时拒绝混合。

`--scores` 复用原 scoring recipe 和冻结 SigLIP，并核验编码器权重。
**得到的是 baseline 的 on-policy 轨迹 importance，不是示范数据原分数**；结果标注了这个限制。
不传此参数则只测风险，不凭空生成 importance。

## 4. 语义难度与风险图

每个状态导出两路相机图片、`identity.json` 和用于重放的前缀/状态。`features_template.csv` 按
`state_uid` 连接数据，包含 `semantic_difficulty,action_error,importance`。
传了 `--scores` 时 importance 自动填入；另外两列不会伪造。

先直接观察 importance 和控制敏感性：

```bash
R=/data1/wyn/prompt-work/control-sensitivity-v1
python -m examples.analysis.curriculum_study risk-plot \
  --results "$R/risk_summary.json" --features "$R/features_template.csv" \
  --output "$R/figures/importance"
```

可选的语义探针：

```bash
python -m examples.analysis.curriculum_study.semantic template \
  --states "$R/states" --output "$R/questions.jsonl"
```

人工填写每个状态的问题、相同数量的候选选项、`correct_index`；检查原始相机方向，必要时显式填写
`rotate_degrees=180`。不要让被评测 VLM 自己生成正确答案，也不要问仅从指令文字就能回答的题。
随后使用一个**完整的本地外部 PaliGemma VLM checkpoint**：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.analysis.curriculum_study.semantic score \
  --model /完整的外部PaliGemma模型目录 \
  --questions "$R/questions.jsonl" --features "$R/features_template.csv" \
  --output "$R/features_semantic.csv"

python -m examples.analysis.curriculum_study risk-plot \
  --results "$R/risk_summary.json" --features "$R/features_semantic.csv" \
  --output "$R/figures/semantic"
```

这里不是已有 `paligemma-3b-pt-224` **仅 tokenizer** 目录，也不是 native PI05 checkpoint。
不会隐式下载模型。使用 suffix 对候选字母做条件似然评分，输出 `-log p(correct option)` 和完整候选分数。
仅用于所选外部模型的语义代理，不自动宣称 π0.5 已理解场景。

`action_error` 只应填同一状态上由**独立专家/示范标签**计算的动作误差。
当前 on-policy rollout 没有这样的标签，所以该列默认留空；不能把 baseline 自己的预测当 ground truth。
该列填写后，`risk-plot` 自动增加对应散点图。

## 5. 测试与验证范围

```bash
python -m pytest -q tests/curriculum_study --confcutdir=tests/curriculum_study
```

测试包括 immutable occurrence pool、episode 隔离、预算和 warmup、原生 PDET 计划读取、原生 scheduler
数值兼容、masked loss、probe 的 RNG/梯度/参数对象保护、真实更新函数的单次调用、LR used/next 区别、
同状态前缀重放、零扰动检查、无夹爪扰动、数据损坏与覆盖保护等。

开发环境没有完整 Transformers/Accelerate、GPU、LIBERO assets/模型。
CPU stand-in 与原生函数体兼容测试通过，不等于已经在 A100 上完成训练或 MuJoCo 全链路验证；
首次上机请先运行 dry-run 和最小 perturb smoke test。可视化结论必须等待真实测量。
