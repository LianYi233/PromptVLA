# Task7：训练后自动评测与时间对应关系诊断

本入口把已有的 **RoPE / Time-Input 两组训练、双种子闭环评测、常量动作检查、单步 Flow 探针，以及从纯噪声开始的完整采样对照** 串在一起。默认保留原来的 Task7 单示教、1000 更新、双 GPU、每卡 batch32、VLM2 + 完整 Action Expert 设置。每个模型训练结束并通过审计后，立即评测该模型；全部完成后自动出图。

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

流程会重新审计现有模型、校验并复用已经完成的闭环评测，然后重新生成常量动作图、单步 Flow 探针图和完整采样对照图。无需重新训练、删除评测缓存或更换 repeat。只有全部机制分析产物生成后，`pipeline_state.json` 才会标记为完成。

## 实际执行顺序

1. 检查并记录固定实验设置。
2. 训练 RoPE，检查最终 `001000/pretrained_model` 和采样/噪声审计记录。
3. 立即评测 RoPE 的全部场景与策略种子组合。
4. 训练 Time-Input，完成相同审计，立即评测其全部组合。
5. 检查两组训练设置与随机流匹配，保存完整配对成功率报告。
6. 读取训练后的时间投影，生成重复同一动作 50 次的输入诊断图。
7. 在完整模型上运行真实动作片段的单步 Flow 探针，保留旧的 `fig04` 机制图。
8. 使用同一个 Time-Input checkpoint，从相同纯噪声完成 Correct / Zero / Reversed 三组原生采样，生成 `fig05`，分别报告前 10 步和完整 50 步。

任一训练、审计、评测或绘图阶段失败，流程停止并把失败阶段写入 `pipeline_state.json`。不会把缺失评测当作失败 episode，也不会输出一个看似完整的最终报告。再次执行同一设置时可显式使用 `--reuse-existing`；修改已冻结的评测设置时使用新的 `--output` 目录。

## 默认评测设置

| 设置 | 默认值 |
| --- | --- |
| 模型 | `rope,time_input` |
| 场景种子 | `20261103,20261104` |
| 策略推理种子 | `1001,1002,1003` |
| 初始状态 | 每个组合固定 `0..9` |
| 平移强度 | `0.12 m` |
| 评测次数 | 每个模型 `2 × 3 × 10 = 60`，两模型共 120 |
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
```

使用 `--output` 指定另一目录。训练 checkpoint 和原始闭环评测仍保留原有目录结构，不由分析目录替代。

## 验证命令

```bash
python -m pytest -q \
  tests/scripts/test_pi05_task7_train_eval.py \
  tests/scripts/test_pi05_task7_full_sampling.py \
  tests/scripts/test_pi05_task7_full_sampling_outputs.py \
  tests/scripts/test_pi05_task7_time_evidence.py \
  tests/scripts/test_pi05_task7_dualseed_evidence.py \
  tests/scripts/test_pi05_task7_constant_action.py \
  tests/scripts/test_pi05_task7_dualseed.py
```

这些 CPU 测试验证真实 PI05 配置的保存/解析、调度顺序、失败停止、checkpoint 选择、阶段采样、时间干预、按示教聚合及数据驱动绘图。配置回归测试使用真实 draccus 解析器，仅替换大模型权重和 tokenizer 加载。完整 3B 模型、数据视频读取、DDP 训练和 MuJoCo 闭环评测仍需要服务器环境。
