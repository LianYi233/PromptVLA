# Task7：训练后自动评测与时间对应关系诊断

本入口把 **RoPE / Time-Input 两组训练、双种子闭环评测、常量动作检查、单步 Flow 探针、从纯噪声开始的完整采样对照、七维动作分析，以及同一 Time-Input checkpoint 的三条件闭环测试** 串在一起。默认保留原来的 Task7 单示教、1000 更新、双 GPU、每卡 batch32、VLM2 + 完整 Action Expert 设置。每个模型训练结束并通过审计后，立即评测该模型；全部完成后自动出图。

## 一条命令启动

在已配置好的 LeRobot 环境中：

```bash
cd /home/wyn/VLAA/PromptVLA

# 只打印执行流程，不要求存在 plan/model/GPU，不写文件。
GPU_IDS=0,1 bash run_task7_train_eval.sh --repeat 2 --dry-run

# 一个新的 run 编号：RoPE 训练→评测，Time-Input 训练→评测，然后机制分析。
GPU_IDS=0,1 bash run_task7_train_eval.sh --repeat 2
```

`--repeat` 是输出 run 的编号，沿用原实验的 P1/D0/E1/T1 随机流。**更换 repeat 不自动等于独立训练种子。** 若该编号已有训练目录，默认拒绝覆盖；选用未使用的编号。

对于已经训练好的 r1：

```bash
# 两个 checkpoint 都必须完整存在，不启动训练；仍自动评测与出图。
GPU_IDS=0,1 bash run_task7_train_eval.sh --repeat 1 --skip-training

# 允许复用完整 checkpoint，并训练尚不存在的那一组。
GPU_IDS=0,1 bash run_task7_train_eval.sh --repeat 1 --reuse-existing
```

复用前会检查训练计划、训练日志中的随机流与模型配置。缺少最终 checkpoint 的不完整训练目录不会被覆盖。已有闭环评测也必须通过原有模型/扰动清单校验才能复用。

### 从机制绘图阶段的配置加载报错恢复

若旧版在 `pi05_task7_time_evidence` 中报出 ``The fields `type` are not valid for PI05Config``，这是离线探针的配置解析入口错误。checkpoint 中的 `"type": "pi05"` 是正常的类型标记；修复版通过 `PreTrainedConfig` 注册表读取，再检查模型类型。不要删除该字段或修改 checkpoint，否则会影响已有评测的指纹校验。

更新分支后，在相同环境和相同参数下重新执行：

```bash
GPU_IDS=0,1 bash run_task7_train_eval.sh --repeat 1 --skip-training
```

流程会重新审计现有模型、校验并复用已经完成的原始双模型闭环评测，然后生成离线机制图，并完成新增的三条件闭环轨迹测试。无需重新训练、删除旧评测缓存或更换 repeat。三条件轨迹测试使用独立缓存，首次运行时 Correct 也要执行。只有全部阶段成功，`pipeline_state.json` 才会标记为完成。

## 实际执行顺序

1. 检查并记录固定实验设置。
2. 训练 RoPE，检查最终 `001000/pretrained_model` 和采样/噪声审计记录。
3. 立即评测 RoPE 的全部场景与策略种子组合。
4. 训练 Time-Input，完成相同审计，立即评测其全部组合。
5. 检查两组训练设置与随机流匹配，保存完整配对成功率报告。
6. 读取训练后的时间投影，生成重复同一动作 50 次的输入诊断图。
7. 在完整模型上运行真实动作片段的单步 Flow 探针，保留旧的 `fig04` 机制图。
8. 使用同一个 Time-Input checkpoint，从相同纯噪声完成 Correct / Zero / Reversed 三组原生采样，生成 `fig05`，分别报告前 10 步和完整 50 步。
9. 直接读取第 8 步的数组，生成七维逐维曲线、误差差值热图、逐条件数据和全部预选窗口图册。该步骤不加载模型。
10. 同一个 Time-Input checkpoint 分别运行 Correct / Zero / Reversed 三组真实闭环测试，记录实际执行动作、末端位姿、夹爪关节状态、物体位姿和官方成功标记，生成配对成功率及实测轨迹图。

任一训练、审计、评测或绘图阶段失败，流程停止并把失败阶段写入 `pipeline_state.json`。不会把缺失评测当作失败 episode，也不会输出一个看似完整的最终报告。再次执行同一设置时可显式使用 `--reuse-existing`；修改已冻结的评测设置时使用新的 `--output` 目录。

## 默认评测设置

| 设置 | 默认值 |
| --- | --- |
| 模型 | `rope,time_input` |
| 场景种子 | `20261103,20261104` |
| 策略推理种子 | `1001,1002,1003` |
| 初始状态 | 每个组合固定 `0..9` |
| 平移强度 | `0.12 m` |
| 原始双模型评测次数 | 每个模型 `2 × 3 × 10 = 60`，两模型共 120，可校验复用旧缓存 |
| 新增三条件闭环次数 | 同一 Time-Input checkpoint 每个条件 60，三条件共 180，独立轨迹缓存 |
| 训练 | 双 GPU，每卡 batch32，1000 更新 |

这套场景沿用之前的 Task7 压力测试设置，可能包含历史校准选择。新流程将其标记为 **`fixed_protocol`**：不依据新 checkpoint 的表现重新筛选场景，也不要求测试时 RoPE 必须落在 40%–50%。它不是未经选择的场景总体测试。

新流程不会修改已有 `calibration-lock.json`。旧双种子校准/比较入口仍保留；已经完成的 `calibrated_stress` 或 `unselected_scenes` JSON 也可以交给新的绘图读取器。

## 完整采样对照：Correct / Zero / Reversed 分别是什么意思？

三组都使用**同一个训练好的 Time-Input 模型、同一份观测和同一份初始纯噪声**，完成 checkpoint 配置规定的全部去噪步骤。模型权重、动作位置顺序和原有 RoPE 保持相同，只改变新增时间分支：

| 条件 | 在每个去噪步骤中进行的操作 | 要检验的问题 |
| --- | --- | --- |
| Correct | 正常使用每个动作位置对应的时间编码 | 正常模型会生成什么动作？ |
| Zero | 关闭整个新增时间残差，保留原有 RoPE | 去掉新增分支是否损害预测？ |
| Reversed | 只反转新增时间编码与 50 个动作位置的对应关系 | 对应关系被打乱后是否损害预测？ |

Zero 使用的仍是 Time-Input 模型的权重，与另行训练的 RoPE 模型不同。Reversed 不反转动作输出、观测、初始噪声或原有 RoPE。三组使用同一份噪声的独立副本，避免前一组的采样过程改变后一组的起点。

**与旧图 A 的区别：** 旧图从 `x_t = t × noise + (1−t) × recorded_action` 做单步动作估计；新图从纯噪声开始调用原生动作采样接口，走完整个去噪过程。记录的未来真实动作只用于评分和画目标曲线，不进入采样起点或模型观测输入。新图的指标是归一化生成动作 MSE，与旧图的 Flow velocity MSE 不能直接比较数值。

当前闭环入口是**预测 50 步、执行前 10 步、再重新观测和规划**。新图标出这 10 步的边界，并单独报告前 10 步与全 50 步的误差；这次离线分析本身不执行仿真动作。`--executed-steps 10` 是分析区间，不能用它修改已有闭环结果的执行协议。

### 直接复用现有 checkpoint，只运行这一步

例如 r2 已经训练完成，tokenizer 位于 `/data/models/paligemma-3b-pt-224`：

```bash
GPU_ID=0 bash run_analyze_task7_full_sampling.sh \
  --repeat 2 \
  --tokenizer /data/models/paligemma-3b-pt-224 \
  --probe-demos 8 --windows-per-demo 3 \
  --noise-seeds 3000,4000,5000 \
  --executed-steps 10 \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/full_sampling
```

这个命令不训练、不重跑闭环评测，也不重跑旧的单步探针。`--repeat` 选择已有模型的 run 编号；如果 checkpoint 来自 r1，就把 `--repeat` 和输出目录中的 `r2` 一起改成 `1` / `r1`。还可以通过 `--time-input-checkpoint /path/to/pretrained_model` 显式指定模型。若训练计划或 checkpoint 根目录不同，传入原来的 `--directory`、`--output-root`。

在同一命令末尾加 `--dry-run`，可先检查采样设置和输出位置。快速检查可使用 `--probe-demos 2 --windows-per-demo 1 --noise-seeds 3000`，并将 `--output` 指向单独的 smoke-test 目录，避免与正式结果混用。

默认最多 **8 条非本次微调训练示教 × 3 个完整窗口 × 3 份初始噪声 = 72 个配对条件**，每个条件执行三次完整采样。示教与窗口按固定规则选择，统计先在每条示教内部对窗口和噪声平均。Flow 去噪步骤来自 checkpoint，不再使用旧探针的 `--flow-times` 网格。

### 如何判断新增时间分支有帮助？

新产物是 `full_sampling/fig05_full_sampling_time_alignment.png` 和 `.svg`，数值保存在 `full_sampling_evidence.json`，逐条件数组保存在 `full_sampling_arrays.npz`。

`Zero − Correct` 和 `Reversed − Correct` 为正，表示干预后动作误差增加；为负，表示干预后误差下降。输出差异只能说明分支影响了模型，不能单独说明正确对应更好。应同时看前 10 步、完整 50 步与不同示教上的配对差值，保留零收益和负收益。完整采样仍是离线预测，不等于机器人执行轨迹或任务成功率；与单条示教动作的误差也不能覆盖所有合理的动作解。

继续使用 `run_task7_train_eval.sh` 时，这一步会自动成为 `full_sampling` 阶段。旧的 `fixed_protocol.json` 和冻结设置保持兼容；`pipeline_state.json` 记录新阶段命令、所用 checkpoint、条件与分析区间，只有四个新产物全部生成才会标记该阶段完成。单独运行分析脚本不会改写原来的 pipeline 状态。

## 七维动作与真实闭环协调：复用现有结果继续实验

原 `fig05` 的 **A 面板只画夹爪，B/C/D 的 MSE 已经平均了全部七维**。新增分析把这七维拆开，避免夹爪事件或某个动作维度掩盖其他维度；它不把七维误差乘以 7 当作另一条有效性证据。真实协调效果由后面的闭环执行检查。

两个独立入口的产物不同：

| 入口 | 所需数据/计算 | 生成的主要图 |
|---|---|---|
| `run_analyze_task7_seven_dim.sh` | 已保存的完整采样 JSON/NPZ，CPU 后处理 | `fig06_seven_dim_actions` 七维曲线、`fig07_seven_dim_error_changes` 逐维误差差值 |
| `run_eval_task7_time_interventions.sh` | 现有 Time-Input checkpoint，三条件 GPU/MuJoCo 闭环 | 全部评测完成后生成 `fig08_closed_loop_success` 成功率、`fig09_execution_trajectories` 实测轨迹 |

**已有完整采样数组时，可以先运行第一步直接出七维图，无需等待第二步闭环完成。** 独立闭环命令不自动调用七维离线分析；完整 `run_task7_train_eval.sh` 已按 `full_sampling → seven_dim_analysis → closed_loop` 顺序集成这两步。闭环入口会在准备及恢复检查前打印图名和标准目录结构下的七维绘图命令，提示命令不会自动运行。

### 第一步：已有数组直接出七维图，不重新推理

```bash
bash run_analyze_task7_seven_dim.sh \
  --input /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/full_sampling \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/seven_dim
```

输入目录必须包含完整的 `full_sampling_evidence.json` 和 `full_sampling_arrays.npz`。脚本核对数组哈希、配对条件及原汇总值，不修改源结果。

若这两个文件尚未生成，可先从已有 r2 checkpoint 补跑完整采样，再执行上面的 CPU 绘图命令：

```bash
GPU_ID=0 bash run_analyze_task7_full_sampling.sh \
  --repeat 2 \
  --tokenizer /data/models/paligemma-3b-pt-224 \
  --probe-demos 8 --windows-per-demo 3 \
  --noise-seeds 3000,4000,5000 \
  --executed-steps 10 \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/full_sampling
```

这一补跑需要 GPU 推理，不重新训练，也不执行 MuJoCo。已有完整数组时无需补跑；如果之前的数组来自 r1，应使用实际 r1 的输入及输出目录，不把 r1 数据改名当作 r2。

- `fig06_seven_dim_actions.png/.svg`：七行共享同一未来动作位置轴，记录目标与三种预测同图，前 10 步有阴影。示例仍按事先固定规则选择。
- `fig07_seven_dim_error_changes.png/.svg`：两张 `7 × 50` 热图，分别为 `Zero − Correct`、`Reversed − Correct`。红色表示干预增加误差，蓝色表示减少误差；白色只说明差值小，不说明绝对误差小。
- `seven_dim_evidence.json`、`seven_dim_metrics.csv`：前 10 步与全 50 步的逐维、前三维、后三个方向控制维度、夹爪、总体指标，包含 MSE、配对差值、示教级区间及输出变化 RMS。
- `seven_dim_position_errors.csv.gz`：全部窗口、噪声、位置、维度的误差贡献及聚合权重。可据此追查原图 demo286 附近的局部红块，不丢弃该示教。
- `all_windows_seven_dim.pdf`：按预先确定的窗口顺序逐页显示，每个窗口固定使用最小噪声种子。统计仍使用全部噪声；没有根据结果挑选有利窗口。

动作图仍使用归一化控制量。维度 0–2 / 3–5 的分组注明标准七维 LIBERO 控制约定，不把旋转控制量随意命名成欧拉角，不把曲线直接解释为米或角度，也不对其累加生成所谓真实末端轨迹。Bootstrap 的单位是示教，窗口和噪声先在示教内平均；区间只描述这个 checkpoint 上的示教采样变化。

### 第二步：一个 checkpoint，三条件原生闭环执行

```bash
GPU_IDS=0,1 bash run_eval_task7_time_interventions.sh \
  --repeat 2 \
  --tokenizer /data/models/paligemma-3b-pt-224 \
  --scene-seeds 20261103,20261104 \
  --test-policy-seeds 1001,1002,1003 \
  --shift-m 0.12 \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/closed_loop
```

两条命令都可加 `--dry-run` 检查计划而不写入结果。第二步不启动训练、不加载独立 RoPE 模型，不重新跑旧单步 Flow 探针。`GPU_IDS=0` 也能顺序执行这个闭环入口；这不改变双 GPU 训练配方。显式 `--time-input-checkpoint` 必须与计划、`--output-root` 和 `--repeat` 指定的最终 Time-Input checkpoint 一致，避免原生评估器意外读取另一个模型。

默认共 **180 个完整 episode**，每条件 60。首次执行会重新跑 Correct，因为旧成功率缓存没有所需轨迹。每个条件、场景种子和策略种子的目录独立；成功率、模型文件、扰动设置、干预配置、实现指纹和十条轨迹全部核验后才复用。中途失败会记录在 `closed_loop_state.json`；用相同命令重跑，已完整完成且核验通过的 cell 可复用，不完整 cell 按原生评估粒度重跑。

#### 从首个 episode 前的输入审计报错恢复

旧版本可能在进度 `0/10` 时报告 `Deployment policy inputs may contain observations and task only`。原生 `PolicyProcessorPipeline` 的转换器会给观测 batch 自动补上 `action=None`、`next.reward=0.0`、`next.done=False`、`next.truncated=False` 和 `info={}`；这些是空占位字段。闭环审计现在接受这些默认值，仍拒绝非空动作标签、非默认结果字段和未知字段，并把原始 batch 原样交给原生 PI05。配对观测指纹仅包含观测和任务。

更新代码后可以直接重跑上面的同一条命令。为避免旧失败尝试的实现指纹再次阻塞，入口仅在**上次状态为 failed、没有完成的 cell、没有任何 episode/task 结果或轨迹证据、其他冻结设置完全相同，且只有源码指纹发生变化**时，自动归档该次失败的配置、原生恢复清单、空进度文件和日志。归档保留在 `empty_failed_attempts/attempt-<unique>/`，包含原相对路径和 `recovery.json`；新 `closed_loop_state.json` 的 `recovered_empty_attempt` 字段记录归档位置。原有锁和无关文件保留。

原生流程会在评测结果目录创建 `.resume.lock`，并在场景配置目录创建 `.fixed_image_noise.lock`；对象平移配置也使用这个共享配置写入锁。恢复器现在识别这两个**具体路径**，先检查能否取得锁，并保留原锁文件的 inode。锁文件存在本身不代表已生成 episode 结果；仍被进程占用的锁、其他未知锁文件和任何完整或部分结果仍会阻止自动恢复。如果旧版本把 `.fixed_image_noise.lock` 报为未知评测产物，更新后可用同一命令重新检查并恢复。

如果已有完整或部分轨迹、已有评测结果、上次进程仍持锁，或修改了 checkpoint/tokenizer/种子/位移等设置，入口不会自动归档继续。源码或实验设置变更且已有结果时，应使用新的 `--output`，保留旧结果供核对；同一实现、同一设置下的正常断点续跑仍遵循原有核验规则。此次修复无需重新训练或修改 checkpoint。

每个 episode 的三种条件使用同一官方初始状态与场景位移，保留原有 `SHA256(policy_seed, init_slot)` 推理随机流。采样仍调用原生 `select_action` / `predict_action_chunk`：每次生成 50 步，执行前 10 步后再次观测，Flow 步数来自 checkpoint。记录器核对对应 chunk 的噪声哈希、去噪起点、完整去噪步骤和实际动作队列。**三种条件动作不同时，后续观测自然会不同；不能继续强行提供同一份观测并称为闭环。**

每条 `episode-000.json` 到 `episode-009.json` 记录：

- 七维模型输出、实际经过后处理送进环境的七维动作、chunk 编号和执行位置。
- 重置并静置后的初始仿真状态指纹；每一步实测末端位置/姿态、夹爪关节状态、任务物体位姿。
- 可用的原始接触信息、官方任务成功标记、终止/超时原因；不把夹爪命令的正负号推断为抓住或释放。
- 初始噪声和观测哈希、原生去噪次数、模型与干预配置等来源信息。

终止步状态在 LIBERO 自动 reset **之前**保存，避免把下一局初始状态错误地画成该局终点。这里只记录用于分析的状态信息，**不声称这些 JSON 是可恢复的完整 MuJoCo/控制器快照**；共同状态短分叉实验仍是后续按需要增加的独立实验。

### 新图如何回答“七维共同作用有没有收益”

`fig08_closed_loop_success` 给出三条件的官方成功/失败配对矩阵、Correct 独赢与干预独赢次数，以及分场景成功率差。主要比较 **Correct vs Zero**，Reversed 作为辅助；保留收益、损失和平局。推理随机种子不当作独立训练种子，固定两场景的结果不包装成对所有场景的统计保证。

`fig09_execution_trajectories` 把三条实际末端轨迹并排显示，标出夹爪命令符号变化时末端所在位置，同时显示实测夹爪关节状态与物体高度变化。姿态信息显示在预定关键帧。这样可检查“手臂到达物体附近时，夹爪命令/实际关节状态怎样变化；物体是否随后抬起”，而不是只看夹爪曲线的跳变是否接近示教第 18 步。命令跳变仍不自动等于抓取、接触或释放事件。

代表 episode 固定为第一个声明的 scene/policy 与初始状态 0，不按成功与否挑选。`all_episode_execution_traces.pdf` 收录全部预声明的配对 episode；`closed_loop_evidence.json` 同时提供逐场景、逐策略种子、逐 cell 的统计与逐 episode 路径。路径长短和动作差异是行为描述，最终是否有帮助仍看官方任务完成结果。

未来完整训练用 `run_task7_train_eval.sh` 时，两步分别自动记录为 `seven_dim_analysis` 和 `closed_loop` 阶段；原冻结设置兼容，新闭环协议另写 `closed_loop_protocol.json`。单独执行以上两个入口不会把旧 `pipeline_state.json` 冒充为已跑过全部新阶段。

## 单步 Flow 机制图：保留原有诊断

`evidence/fig04_time_alignment_evidence.png` 和 `.svg` 是单步 Flow 探针图。所有曲线和色块来自 checkpoint 的实际预测，没有预设 Correct 必须获胜。

### 固定输入与四种条件

使用同一批非训练示教中的完整 50 步窗口。对每个窗口、Flow timestep 和噪声种子，保持观测、动作标签、噪声及原有 RoPE 相同：

- **RoPE**：独立训练的基线模型。
- **Correct**：Time-Input 的正常时间对应关系。
- **Zero**：同一个 Time-Input 模型，关闭整个额外时间残差。
- **Reversed**：同一个 Time-Input 模型，只反转 50 个时间特征与动作位置的对应关系。

Zero 不等于“把物理时间都设为 0”，也不等于独立训练的 RoPE。Reversed 不反转动作标签、观测、噪声或 RoPE。

### 默认覆盖范围

```text
8 条非训练示教 × 每条前/中/后 3 个窗口
× Flow timestep {0.2, 0.5, 0.8}
× 噪声种子 {3000, 4000, 5000}
```

最多 24 个动作窗口、216 个配对输入条件，每个条件比较四种预测。若可用示教或完整窗口更少，输出记录实际覆盖数量。窗口不能跨越 episode 边界，训练使用的那条示教始终排除。

每条示教内部先对窗口、噪声、Flow 时间求平均，再比较不同示教。图中的一条示教不会因为被重复采样多次而被当作多个独立实验。

### 如何看四个面板

1. **预先固定窗口的夹爪动作示例**：展示真实目标与四种预测。动作由 `clean_hat = x_t - t * predicted_velocity` 重构；坐标是归一化动作值，不是物理开合距离。示例按预先固定的顺序选择，不根据哪个方法获胜挑选。
2. **每条示教的预测误差**：显示四种条件的 Flow velocity MSE，便于看改善是否分散在多条示教。
3. **干预造成的误差变化热图**：展示 `Zero − Correct` 与 `Reversed − Correct`，保留正值、零值与负值，观察影响出现在哪些动作位置。
4. **每条示教的配对误差差值**：`Zero − Correct`、`Reversed − Correct` 为正表示正确时间对应关系更好；`Correct − RoPE` 为负表示 Time-Input 优于独立基线。

只有当 Correct 在真实动作片段上比 Zero/Reversed 更好时，这些结果才支持模型利用了正确对应关系。单纯出现不同的输入 embedding 或预测变化，不等于预测更准确。干预也引入训练/测试特征差异，因此它是依赖关系的证据，仍需结合闭环成功率判断收益。

本实验固定 FPS。不能仅凭该图声称掌握了不同执行速度下的真实物理时间。探针目前明确支持原始 `index` 时间基底；其他基底必须先实现正确的时间间隔传递。

## 单独运行机制分析

```bash
GPU_ID=0 bash run_analyze_task7_time_evidence.sh \
  --mode probe --repeat 1 \
  --probe-demos 8 --windows-per-demo 3 \
  --flow-times 0.2,0.5,0.8 \
  --noise-seeds 3000,4000,5000 \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-time-alignment-r1
```

首次快速检查可以缩小为 `--probe-demos 2 --windows-per-demo 3 --flow-time 0.5 --noise-seed 3000`。单数参数与对应复数参数不能同时使用；这只是运行检查，正式对照应使用预先确定的完整网格。

导入已有双种子比较 JSON，只做 CPU 结果绘图：

```bash
bash run_analyze_task7_time_evidence.sh \
  --mode success \
  --comparison-json /path/to/comparison-calibrated_stress-XXXX.json \
  --output /data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-dualseed-existing
```

`--mode all --comparison-json ...` 会先绘制该比较结果，再运行机制探针。旧四组单种子结果仍可用原来的 `--seeds` 模式读取，但不要把新双种子结果套进旧路径。

## 输出位置

默认位于：

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-train-eval/r2/
    fixed_protocol.json
    pipeline_state.json
    comparison-fixed_protocol.json
    comparison-fixed_protocol.csv
    logs/
    constant_action/
        fig00_constant_action_50.png
        fig00_constant_action_50.svg
        constant_action_50_summary.json
    evidence/
        fig01_paired_task7_outcomes.png
        paired_success.json
        fig02_learned_time_residual.png
        fig03_time_input_counterfactual.png
        fig04_time_alignment_evidence.png
        fig04_time_alignment_evidence.svg
        time_alignment_evidence.json
        offline_probe.json
        offline_probe_arrays.npz
    full_sampling/
        fig05_full_sampling_time_alignment.png
        fig05_full_sampling_time_alignment.svg
        full_sampling_evidence.json
        full_sampling_arrays.npz
    seven_dim/
        seven_dim_evidence.json
        seven_dim_metrics.csv
        seven_dim_position_errors.csv.gz
        fig06_seven_dim_actions.png
        fig07_seven_dim_error_changes.png
        all_windows_seven_dim.pdf
    closed_loop/
        closed_loop_protocol.json
        closed_loop_state.json
        comparison-time-interventions.json
        closed_loop_evidence.json
        closed_loop_paired_outcomes.csv
        fig08_closed_loop_success.png
        fig09_execution_trajectories.png
        all_episode_execution_traces.pdf
        evaluations/{correct,zero,reversed}/...
            resume_manifest.json
            eval_info.json
            trajectories/episode-000.json ... episode-009.json
```

使用 `--output` 指定另一目录。训练 checkpoint 和原始闭环评测仍保留原有目录结构，不由分析目录替代。

## 验证命令

```bash
python -m pytest -q \
  tests/scripts/test_pi05_task7_train_eval.py \
  tests/scripts/test_pi05_task7_full_sampling.py \
  tests/scripts/test_pi05_task7_full_sampling_outputs.py \
  tests/scripts/test_pi05_task7_seven_dim.py \
  tests/scripts/test_pi05_task7_closed_loop.py \
  tests/scripts/test_pi05_task7_closed_loop_runtime.py \
  tests/scripts/test_pi05_task7_closed_loop_report.py \
  tests/scripts/test_pi05_task7_time_evidence.py \
  tests/scripts/test_pi05_task7_dualseed_evidence.py \
  tests/scripts/test_pi05_task7_constant_action.py \
  tests/scripts/test_pi05_task7_dualseed.py
```

这些 CPU 测试验证真实 PI05 配置的保存/解析、调度顺序、失败停止、checkpoint 选择、原生采样/动作队列、时间干预、终止状态保留、配对初始状态与噪声审计、逐维误差/示教聚合及数据驱动绘图。输入审计回归实际调用仓库的 `batch_to_transition` / `transition_to_batch`，覆盖原生预处理产生的空元数据，并核对动作和 RNG 与未加审计的原生路径逐位一致；恢复测试同时覆盖空失败归档和存在部分/完整结果时拒绝混用。配置回归测试使用真实 draccus 解析器，重量级模型、tokenizer 和仿真器在相关测试中替换为小型测试对象。完整 3B 模型、数据视频读取、DDP 训练和 MuJoCo 闭环评测仍需要服务器环境。
