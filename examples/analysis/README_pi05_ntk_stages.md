# PI05 四时间点 NTK 分析

比较 **VLM 与 action expert** 在训练前、750 步 priming 后、1000 步 bridge（论文 stage 2）后、最终训练后的局部敏感性。
`backbone` 与 `prompts` 分开绘图，默认两者都测。这里 action backbone 包含 Gemma expert、action 输入/输出投影和 time MLP；不包含只用于 inpainting 的 role embedding。VLM 包含完整 PaliGemma，包括视觉编码器。参数按对象去重，未参与输出的参数贡献零梯度，但仍计入该组参数总数。

## 1. 先区分累计步数与 flow 步数

当前 `train_pi05_prompt_ablation.sh full_reference` 的默认配方是 **750 + 250 + 3000 = 4000** 次更新。
前 1000 次更新位于 `RUN_DIR_next_action_pretrain`，后续 flow 在 `RUN_DIR` 中从 0 重新计数。

| 时间点 | 累计更新数 | checkpoint |
| --- | ---: | --- |
| 训练前（包含本次训练实际初始化的 dual prompts） | 0 | `RUN_DIR_next_action_pretrain/checkpoints/000000/pretrained_model` |
| Priming 完成，尚未执行 bridge 更新 | 750 | `RUN_DIR_next_action_pretrain/checkpoints/000750/pretrained_model` |
| Bridge / 论文 stage 2 完成 | 1000 | `RUN_DIR_next_action_pretrain/checkpoints/001000/pretrained_model` |
| 整个训练累计 3000 步 | 3000 | `RUN_DIR/checkpoints/002000/pretrained_model` |
| 已有默认配方的 flow 3000 步完成 | 4000 | `RUN_DIR/checkpoints/003000/pretrained_model` |

分析默认使用表中累计 3000 的版本。如果“3000 steps”指原脚本的 **flow 3000**，设置 `FINAL_FLOW_STEPS=3000`；图上会如实标成累计 4000。
不修改现有消融训练的默认步数。`full_reference` 现在默认开启阶段快照，并每 1000 个 flow 更新保存一次，
因此一次默认训练会同时保留累计 3000 的 `002000` 和最终 flow 3000 的 `003000`。

## 2. 训练时保存准确的四个快照

现有脚本原本只保存预训练结束的 1000 checkpoint，`SAVE_FREQ=750` 也不会改变其独立预训练保存频率。
现在直接运行 `full_reference` 就会自动传入 `--policy.ntk_save_stage_snapshots=true`，
额外保存每个训练阶段自己的初始 checkpoint，以及 priming 结束的 750 checkpoint。
其他 variant 默认不启用阶段快照，flow 默认仍每 3000 步保存。可用 `NTK_SAVE_STAGE_SNAPSHOTS=true/false`
控制阶段快照，`SAVE_FREQ` 覆盖 flow 保存间隔。所有快照保存完整模型、实际 prompts、预处理器和训练状态，
额外占用与完整 checkpoint 相当的空间。

重新跑原来的 full_reference 配方并自动保存分析所需模型（换一个未使用的 `RUN_NAME`）：

```bash
RUN_NAME=pi05-full-reference-ntk-seed0 GPU_IDS=0 \
  bash examples/training/train_pi05_prompt_ablation.sh full_reference 0
```

该命令沿用下述环境变量所指定的模型、数据和 tokenizer 路径；默认配方为 750 + 250 + 3000 flow。
会保留预训练目录的 `000000 / 000750 / 001000`，以及 flow 目录的 `000000 / 001000 / 002000 / 003000`。
其中 flow `000000` 与预训练 `001000` 是同一个阶段边界；整个实验的训练前模型应取预训练目录的 `000000`。

累计训练 3000 步的新实验（先在当前训练环境 `pip install -e .`）：

```bash
export DATASET_REPO_ID=libero
export DATASET_ROOT=/root/autodl-tmp/datasets/libero
export PRETRAINED_PATH=/root/autodl-tmp/models/pi05_libero_base
export TOKENIZER_PATH=/root/autodl-tmp/models/google/paligemma-3b-pt-224
export GPU_IDS=0
export RUN_NAME=pi05-full-reference-ntk-total3000-seed0

FLOW_STEPS=2000 SAVE_FREQ=2000 \
  bash examples/training/train_pi05_prompt_ablation.sh full_reference 0 \
  --policy.ntk_save_stage_snapshots=true
```

若要沿用已有 **750 + 250 + 3000** 配方，改为 `FLOW_STEPS=3000 SAVE_FREQ=1000`，同时保留累计 3000 与最终 flow 3000 的 checkpoint。
训练结束后，完整 flow 阶段的 `checkpoints/000000` 对应累计 1000，**不能**当作整个实验训练前的快照。
如果旧实验没有保存 0 或 750，无法从 1000/3000 的权重反推出这些时间点；需要重新运行并打开上述快照开关。脚本不会用随机初始化的 prompts 或别的 checkpoint 代替缺失时间点。

## 3. 四时间点分析

在保存了训练权重和数据的机器上运行。无需 LIBERO 模拟器，也不会执行机器人动作。

```bash
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-total3000-seed0

# 先检查时间点、文件和保存的训练步数；不会加载模型或使用 GPU。
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0 --dry-run

# 正式分析：32 个固定 samples，10 个配对随机种子，每种子 4 个输出探针。
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0
```

已有 flow-3000 的实验：

```bash
FINAL_FLOW_STEPS=3000 \
  bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0
```

### 训练尚未结束：先分析前三个时间点

只要预训练目录中 `000000 / 000750 / 001000` 三个 checkpoint 已完整保存，就可以先画训练前、
priming 后和 stage 2 后的散点图与变化趋势，无需最终 flow checkpoint：

```bash
# RUN_DIR 是本次实验的正式 flow 输出目录，不带 _next_action_pretrain 后缀。
# 下面的 1 是示例空闲 GPU 编号，请按实际机器修改。
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 1 --through-stage=stage2 --dry-run
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 1 --through-stage=stage2
```

图片自动保存到 `$RUN_DIR/ntk_stages`（或 `NTK_OUTPUT_DIR`）。阶段散点图此时为三个面板，
轨迹和指标趋势图也只包含累计 0、750、1000 步。支持 `--through-stage=before/priming/stage2/final`，
默认 `final` 仍检查全部四个时间点，不会自动跳过缺失的 750 模型。

训练结束后，保持相同数据、样本、种子、scope、估计参数和输出目录，去掉 `--through-stage=stage2`：

```bash
# 默认 full_reference 的最终 flow checkpoint 为 003000，累计更新 4000。
FINAL_FLOW_STEPS=3000 \
  bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 1
```

脚本会复用前三个时间点已保存的结果，只计算最后一个时间点，并更新同名图片为四阶段版本。
若目标是累计 3000 步，使用 `FINAL_FLOW_STEPS=2000`。完整 backbone NTK 需要额外显存，
训练同时进行时请使用另一张空闲且显存充足的 GPU。分析过程不会更新训练权重。

也支持独立指定四个路径：

```bash
python -m lerobot.scripts.analyze_pi05_ntk_stages \
  --before=/path/to/pretrain/checkpoints/000000 \
  --priming=/path/to/pretrain/checkpoints/000750 \
  --stage2=/path/to/pretrain/checkpoints/001000 \
  --final=/path/to/flow/checkpoints/002000 \
  --final-flow-steps=2000 \
  --dataset-repo-id=libero --dataset-root=/path/to/libero \
  --tokenizer-path=/path/to/paligemma-3b-pt-224 \
  --output-dir=outputs/ntk_stages
```

`--scope=backbone` 只画 backbone；`--scope=prompts` 只画实际微调参数，显存需求更低。
默认 `--scope=both`。完整 backbone 分析会临时启用数十亿参数的梯度；建议先在空闲 A100 80GB 上用
`--num-samples=2 --seeds=0 --output-probes=1 --output-dir=outputs/ntk_smoke` 做一次小规模运行。
这不是完整实验结果；正式运行用另一个输出目录。没有执行 optimizer step，也不修改 checkpoint。

默认使用前 32 个样本，保持旧实验固定样本的习惯。若希望覆盖更多任务/episode，使用
`--indices=0,500,1000,...` 指定真实有效索引（不要把省略号写进命令）。索引会保存到 manifest；四个时间点必须一致。
默认未进行图像增强。`TOKENIZER_PATH` 可修复跨机器迁移后保存的 tokenizer 路径失效问题。

## 4. 指标与可比性

所有时间点均测同一个**完整视觉语言条件下的 flow velocity prediction**，不是训练 loss 的梯度，也不是多步积分后的最终控制动作。
即使 750 checkpoint 来自纯动作 priming，分析时也调用 `predict_velocity`，不调用其纯动作 inpainting objective。
否则 VLM 在 priming 的计算图中不存在，会把结构性零误画成敏感性下降。

设第 i 个样本的有效 action velocity 向量为 f_i，参数组 m 的 Jacobian 为 J_i,m。
目标是输出坐标求和后的 sample Gram：

```text
K_m[i,j] = sum_a < d f_i[a] / d theta_m, d f_j[a] / d theta_m >
p_k = lambda_k(K_m) / sum_j lambda_j(K_m)
Spectral effective rank = exp(-sum_k p_k log p_k)
Parameter-normalized tangent energy = Tr(K_m) / P_m
```

横轴采用谱熵有效秩，不是 participation rank；后者额外保存到 CSV。纵轴 **没有再除以 effective rank，也没有除以样本数**。
不同样本数、输出维度/长度或参数组定义的绝对能量不能直接与旧图比较。零 kernel 的秩和能量定义为 0；实测任一组全零会报错，避免静默接受断开的计算图。

完整输出 Jacobian 太大，脚本使用共享 Rademacher 输出探针，近似上述 NTK：

```text
g_i,m,q = J_i,m^T r_q,  E[r_q r_q^T] = I
Khat_m[i,j] = mean_q <g_i,m,q, g_j,m,q>
```

每个探针必须在**所有样本、四个 checkpoint** 间共享；每个配对 seed 固定其 flow noise、timestep 和输出探针。
噪声/timestep 按样本固定，在 checkpoint 切换、模型初始化或断点续跑时不会漂移。
默认 10 个 seed 衡量探针与 flow 随机性的波动，**不是 10 次独立训练的误差条**。

对 backbone，为避免保存 N × P 的巨大梯度矩阵，谱计算再用固定的 **8192 维 CountSketch** 压缩参数轴；因此 backbone 的 rank 是近似值。
**能量从压缩前的梯度平方范数计算**，不受参数 sketch 的误差影响（仍有输出探针误差）。
Prompt 参数较少，保留完整参数轴，只有输出投影近似。
JSON 保存 kernel、谱、参数量、参数名称/形状、sketch trace 误差。正式用于论文前，建议固定所有其他设置，增大
`--output-probes` 和 `--sketch-dim` 检查主要趋势是否稳定；换目录保存，避免混用估计口径。

分析强制 float32、关闭 compile、梯度 checkpointing 和 frozen-observation detach 优化。
先调用 `eval()`，再临时设置目标参数的 `requires_grad=True`，避免 PI05 的 eval/train 钩子重新冻结 backbone。
四次均使用训练前保存的同一个预处理器（包括归一化统计），并核对模型结构配置、prompt 权重存在性和实际训练步数。
所有 checkpoint 应来自同一次训练；文件元数据无法证明人为搬移的权重来源，需保持实验目录的对应关系。

冻结 backbone 的参数值可以不变，但其 NTK 仍会随训练后的 prompts 和内部激活改变。
这些图描述局部输出敏感性；不能直接解释为参数实际更新量、性能提升，或预设能量差一定缩小。

## 5. 输出与断点续跑

每个时间点/seed 完成后原子保存一次 JSON；中断后用同一命令继续。
manifest 记录数据样本摘要、checkpoint 元数据、种子、估计器和源代码摘要；允许在相同协议下追加后续时间点，
其他配置变化会要求使用新输出目录。

每个 scope 输出以下 PNG 和可编辑矢量 PDF：

- `backbone_stages` / `prompts_stages`：所选时间点的散点图，共用坐标轴；四阶段为 2×2，前三阶段为 1×3。
- `backbone_trajectory` / `prompts_trajectory`：保留原横纵轴的所选阶段中位数轨迹图。
- `backbone_effective_rank` / `prompts_effective_rank`：有效秩随累计步数的变化。
- `backbone_tangent_energy` / `prompts_tangent_energy`：参数归一化能量随累计步数的变化。
- `similarity/{scope}_stage_similarity`：VLM 与 Action head 的跨阶段 CKA 热图（至少两个阶段且保存了 kernel 时自动生成，详见第 8 节）。

蓝色为 VLM，橙色为 action expert；正能量使用 log 纵轴；误差条/阴影为配对 seed 的中位数和 IQR。
另有 `metrics.csv`、`results.json` 和 `manifest.json`。

重新绘图无需 GPU：

```bash
python -m lerobot.scripts.plot_pi05_ntk_stages "$RUN_DIR/ntk_stages/results.json"
```

依赖沿用 PI05 训练环境，并安装 `matplotlib`。真实权重和数据只在用户训练机器上可用；提交时的 CPU 测试验证解析 Jacobian、估计器、快照步数解析与绘图，不代替真实 GPU 分析。

## 6. 仅重绘：上排四阶段 NTK，下排 loss 曲线

已有四阶段 `results.json` 后，下面的命令不加载模型、数据集或 PyTorch，不计算 NTK，不需要 GPU。
只依赖 NumPy 和 Matplotlib。上排为四个共用坐标范围的散点图，下排为累计训练步数上的 loss；
四根箭头分别指向 0、750、1000 和最终 checkpoint 的累计步数（默认 full_reference 为 4000）。

```bash
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-seed0
bash examples/analysis/replot_pi05_ntk_with_loss.sh "$RUN_DIR"
```

默认读取 `$RUN_DIR/ntk_stages/results.json`，以及
`/root/autodl-tmp/logs/prompt-ablation/pi05-full-reference-ntk-seed0.log`。
可用 `TRAIN_LOG=/actual/path/run.log` 或 `--training-log=/actual/path/run.log` 指定日志，
用 `NTK_RESULTS=/actual/path/results.json` 指定 NTK 结果。
`--scope=backbone` 可只输出 backbone 图；默认同时绘制 backbone 和 prompts。

新图保存到 `$RUN_DIR/ntk_stages/with_loss/`：

- `backbone_ntk_with_loss.png/pdf/svg` 和 `prompts_ntk_with_loss.png/pdf/svg`。
- `loss_curve.csv`：整理后的累计步数、loss 与训练阶段。
- `plot_metadata.json`：输入来源、文件摘要及绘图参数。

原始 NTK 结果和图片保持不变。若需要额外平滑，可加 `--smooth-window=5`，即对每个训练目标内
连续 5 个已记录的 loss 做尾随均值，并保留浅色原始曲线；默认不额外平滑。

### 本地日志与 W&B 的步数区别

当前训练器在前 1000 步预训练中关闭 W&B，因此该配方的 W&B flow run 通常只覆盖累计 1000–4000 步。
启动脚本通过 `tee` 保存的完整 `.log` 则包含 priming、bridge 和 flow。
重绘时，预训练 local step 保持不变，flow local step 加 1000。
训练器会将 1200/1400 等步数取整显示为 `1K`，脚本依据配置中的 `log_freq` 和完整、连续的记录恢复精确步数，
不能把日志中的 `1K` 直接当作每条记录的真实更新数。缺失或拼接的日志可能无法恢复；此时请改用精确步数 CSV。
若日志没有保存 `log_freq`，可显式传入当时真实的 `--log-freq=200`。

loss 是训练日志保存的窗口均值；默认每 200 步一条，并且文本仅保存三位小数。
跨越 750 步目标切换的窗口（例如 601–800）单独画为灰色叉号，不与相邻目标连线或一起平滑。
step 0 通常没有 loss，750 也不一定正好有记录；不会为它们插值或伪造 loss。
箭头指示 checkpoint 的时间位置，不声称该位置存在 loss 观测。不同阶段的训练目标也不完全相同。

### 使用 W&B 导出的 CSV

需要具体 run 导出的数据，`https://wandb.ai/home` 本身不包含可直接读取的某次实验历史。
导出一个 run 的 `train/loss`，优先选择 `train/steps` 为横轴；脚本也识别 `Step`、`_step`。
若 CSV 只包含 flow 的 local 0–3000 步，明确加上偏移：

```bash
bash examples/analysis/replot_pi05_ntk_with_loss.sh "$RUN_DIR" \
  --loss-csv=/path/to/flow_loss.csv --step-offset=1000
```

此时前 1000 步没有 loss 数据的部分会留空，NTK 四面板与箭头仍保留。若 CSV 已经采用累计步数则不加偏移。
可用 `--step-column='Step' --loss-column='run-name - train/loss'` 明确列名；导出了多个 run 时必须选择其中一个。
也可提供完整的两列 `cumulative_step,loss` CSV。脚本不登录 W&B，也不会上传任何训练数据。

## 7. 无 loss 数据：四图美化与可编辑 PPT 示意图

已有四阶段 NTK 结果时，只重新绘制散点图，不需要训练日志、loss CSV、模型或 GPU：

```bash
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-seed0
bash examples/analysis/replot_pi05_ntk_panels.sh "$RUN_DIR"
```

输出目录为 `$RUN_DIR/ntk_stages/panels/`。每个 scope 保存四张独立面板
`{scope}_before / priming / stage2 / final`，以及一行四图 `{scope}_four_stages`，
均导出 PNG、PDF、SVG。蓝色圆点对应 VLM，橙色菱形对应 action expert；
淡色点为配对 seed，实心标记和误差线为中位数与 IQR。四阶段使用相同坐标范围，保留真实指标值。

下载随本次任务提供的 `PrimingVLA-NTK-Loss-Editable-v3.pptx` 并放到训练机器后，
可以一次生成图片并自动回填 PPT：

```bash
bash examples/analysis/replot_pi05_ntk_panels.sh "$RUN_DIR" \
  --pptx-template=/root/autodl-tmp/PrimingVLA-NTK-Loss-Editable-v3.pptx
```

完整结果保存在 `$RUN_DIR/ntk_stages/panels/PrimingVLA-NTK-Loss-Filled.pptx`，
含 backbone 和 prompts 两页。上方为真实 NTK 图，下方为用户指定的 **loss 示意曲线**，
不是训练记录或对训练 loss 的估计。示意曲线不标定数值，并在图中明确标注为 schematic。
PPT 的 loss 曲线、阶段箭头、文字与分区为原生可编辑对象；loss 曲线可右键“编辑顶点”。
NTK 面板作为高分辨率图片插入，同时保留单独的 SVG/PDF 矢量文件。

模板上方在回填前只有待填图位置，没有虚构 NTK 点；脚本会删除待填文字并替换图片。
原模板和 `results.json` 均保持不变。模板明确对应累计 0、750、1000、4000 步；
如果已有 NTK 结果最后是累计 3000，脚本会拒绝将它改标为 4000。

## 8. 仅重绘：跨阶段 NTK 相似度热图

每个 scope 输出 **两张热图**：VLM 一张、Action head 一张，并额外保存两图并排的组合图。
每张完整结果为 **4×4**；横纵坐标均是 checkpoint 的累计步数 **0、750、1000、4000**，
每个格子比较两个 checkpoint 的 NTK 结构，不是“每阶段各画一张样本矩阵”。
坐标从现有 manifest 读取；若最后是累计 3000，会如实显示 3000。

已有 `results.json` 后，在训练环境内直接运行；只需要 NumPy、Matplotlib，无需模型、数据集、loss 或 GPU：

```bash
git pull --ff-only origin prompt-ablation
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-seed0
bash examples/analysis/replot_pi05_ntk_similarity.sh "$RUN_DIR"
```

默认自动绘制结果中已有的 backbone / prompts scopes。只画 backbone：

```bash
bash examples/analysis/replot_pi05_ntk_similarity.sh "$RUN_DIR" --scope=backbone
```

默认只显示下三角（含对角线），避免重复信息；`--full-matrix` 显示完整对称矩阵。
两张图均采用相同的 **0–1 红橙色阶**：低值为浅橙，高值为深红，无背景网格。
格子内数字根据背景亮度自动使用黑色或白色，保证橙色中间值与深红色高值都清晰可读。
如需调整配色，修改 `src/lerobot/scripts/plot_pi05_ntk_similarity.py` 顶部的 `HEATMAP_COLORS`：

```python
HEATMAP_COLORS = ("#FFF4E8", "#FED7AA", "#FB923C", "#E34A33", "#A50F15")
```

格子标注中位数，保留两位小数；完整精度和四分位数保存在 JSON 中。
支持仅有前两个或前三个阶段的结果，分别生成 2×2、3×3 矩阵。

默认输出到 `$RUN_DIR/ntk_stages/similarity/`：

- `{scope}_stage_similarity.png/pdf/svg`：VLM、Action head 并排，共用色条。
- `{scope}_vlm_stage_similarity.png/pdf/svg`、`{scope}_action_stage_similarity.png/pdf/svg`：独立图片。
- `stage_similarity_pairs.csv`：每个 scope/module、配对 probe seed、不同阶段对的原始 CKA。
- `stage_similarity.json`：中位数、25%/75% 分位数、每个 seed 的矩阵、样本顺序、输入文件摘要与绘图设置。

`scope` 为 `backbone` 或 `prompts`。PDF/SVG 保留矢量文字和绘图元素。
可用 `--output-dir=/path/to/figures` 改输出目录，`NTK_RESULTS=/path/to/results.json` 改输入文件。
原来的散点图和 `results.json` 不会被这个重画命令改动。
以后运行原来的 NTK 分析或 `plot_pi05_ntk_stages`，存在至少两个阶段的完整 kernel 时，
也会自动生成 `similarity/` 下的图；仅有 scalar metrics 的旧结果仍能画原来的图。

### 比较的具体含义

对同一个模块、同一个 probe seed，令两个 checkpoint 的样本 NTK 为 K_s、K_t，计算：

```text
H = I - 11^T / N
A_s = H K_s H
CKA(s,t) = <A_s, A_t>_F / (||A_s||_F ||A_t||_F)
```

先在**相同 seed** 内计算每个阶段对的 CKA，再跨 seed 取中位数与 IQR；
不混配不同 seed，也不先平均 kernel 再计算相似度。
这里使用通常的 biased-HSIC normalization，合法非退化 PSD kernel 的 CKA 位于 0–1。
输入必须来自同一个 shared-probe 分析 manifest，保持样本及顺序、输出探针、noise/timestep、
参数组和 sketch 设置一致。脚本检查阶段/seed 完整性、步数、样本维数、kernel 的有限性/对称性/PSD，
并在保存了参数签名时检查签名一致；不要手工合并不同协议的 JSON。
零矩阵或中心化后为零的常数矩阵，其 CKA 未定义，脚本会报出对应阶段与模块，不会填成 1。

相似度消除了整体幅度缩放，补充原图的 tangent energy 和 effective rank：
即使秩和能量相近，样本间的梯度响应关系仍可能不同。
0↔750、750↔1000、1000↔4000 分别对应 priming、bridge、adaptation 的结构变化；
0↔4000 表示相对初始状态的整体变化。高相似度表示当前固定样本上的中心化梯度结构相近，
不直接证明语义能力保持、任务成功率提升或参数未更新。
即使 backbone 冻结，训练后的 prompts 和其他模块也可能改变其 NTK。

backbone 的 CKA 继承 CountSketch 与输出探针近似，prompts 的 CKA 继承输出探针近似。
IQR 反映 probe/flow 随机性，**不是多次独立训练的不确定性**。
方法参考 [Kornblith et al., ICML 2019, Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html)；这里将核对齐用于跨 checkpoint 的 NTK 比较。
