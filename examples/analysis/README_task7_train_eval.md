# Task7：训练后自动评测与时间对应关系诊断

本入口把已有的 **RoPE / Time-Input 两组训练、双种子闭环评测、常量动作检查、真实动作反事实探针和绘图** 串在一起。默认保留原来的 Task7 单示教、1000 更新、双 GPU、每卡 batch32、VLM2 + 完整 Action Expert 设置。每个模型训练结束并通过审计后，立即评测该模型；全部完成后自动出图。

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

## 实际执行顺序

1. 检查并记录固定实验设置。
2. 训练 RoPE，检查最终 `001000/pretrained_model` 和采样/噪声审计记录。
3. 立即评测 RoPE 的全部场景与策略种子组合。
4. 训练 Time-Input，完成相同审计，立即评测其全部组合。
5. 检查两组训练设置与随机流匹配，保存完整配对成功率报告。
6. 读取训练后的时间投影，生成重复同一动作 50 次的输入诊断图。
7. 在完整模型上运行真实动作片段的 Correct / Zero / Reversed 探针，生成主要机制图。

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

## 主要机制图：模型是否利用正确的时间对应关系

`evidence/fig04_time_alignment_evidence.png` 和 `.svg` 是主要图。所有曲线和色块来自 checkpoint 的实际预测，没有预设 Correct 必须获胜。

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
```

使用 `--output` 指定另一目录。训练 checkpoint 和原始闭环评测仍保留原有目录结构，不由分析目录替代。

## 验证命令

```bash
python -m pytest -q \
  tests/scripts/test_pi05_task7_train_eval.py \
  tests/scripts/test_pi05_task7_time_evidence.py \
  tests/scripts/test_pi05_task7_dualseed_evidence.py \
  tests/scripts/test_pi05_task7_constant_action.py \
  tests/scripts/test_pi05_task7_dualseed.py
```

这些 CPU 测试验证调度顺序、失败停止、checkpoint 选择、阶段采样、时间干预、按示教聚合及数据驱动绘图。完整 3B 模型、数据视频读取、DDP 训练和 MuJoCo 闭环评测仍需要服务器环境。
