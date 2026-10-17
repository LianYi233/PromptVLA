# VLM-only training with fixed-probe effective-rank monitoring

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
GPU_IDS=0,1 BATCH_SIZE=32 \
VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" \
bash examples/training/train_pi05_vlm_only_rank_sweep.sh 1 \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

This sequentially trains `pi05-base-vlm{1,2,4,8,16,32}-seed1` from the same base
checkpoint. Both GPUs train each model with batch 32 **per GPU** (global batch 64).
The existing base QUANTILES processors, learning-rate schedule and frozen-backbone
VLM-only setup are retained. Input models/tokenizer default to `/home/wyn/models/`,
the dataset to `/home/wyn/datasets/libero`, checkpoints to
`/data1/wyn/chkpt/2601-lerobot/PromptVLA`, and logs to `/data1/wyn/chkpt/2601-lerobot/logs`.
Only the final **full** checkpoint, `checkpoints/003000`, is saved. The diagnostic
also saves tiny prompt-only snapshots; it never saves intermediate backbone weights.
`torch.compile` is disabled. Existing run directories cause an error before training.
To restart a partial sweep, explicitly set the remaining `VLM_PROMPT_TOKEN_COUNTS`.

## Swap initialization across training seeds / 跨服务器复用初始化

The seed number alone does not identify the exact initial prompt: model construction
consumes random draws before prompt initialization. The existing monitor saves the
actual initial values in `prompt_rank/step_000000.npz`, key `prompt`. Do not use
`step_003000.npz` or final model weights for this experiment.

先在两台机器使用同一 PromptVLA 代码版本；以下 `/path/to/prompt-inits` 是占位路径，需替换为实际目录。源服务器导出实际 seed1 初始 prompt：

```bash
cd /home/wyn/VLAA/PromptVLA
python -m lerobot.scripts.export_pi05_prompt_init export \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-base-vlm2-seed1 \
  --expect-seed 1 \
  --output /path/to/prompt-inits/pi05-base-vlm2-init-seed1.safetensors
```

This reads the step-zero NPZ, probe manifest and a saved `train_config.json`.
It validates the source training seed and shape, records provenance/checksums, and
exports only the `[2,2048]` float32 prompt (16 KiB tensor payload plus metadata).
The completed 3000-step checkpoint is used only to read training configuration;
its learned prompt is **never** exported. If step zero is missing, export fails
instead of silently substituting trained weights. Existing export files are not overwritten.

把这一份 `.safetensors` 文件下载后上传到另一台服务器，放在实际的 prompt 初始化目录。
无需复制完整模型或优化器。两台机器可以互相 SSH 访问时，也可在源服务器使用：

```bash
scp -P <目标SSH端口> \
  /path/to/prompt-inits/pi05-base-vlm2-init-seed1.safetensors \
  <用户名>@<目标服务器地址>:/path/to/prompt-inits/
```

目标服务器验证文件，然后训练：

```bash
cd /home/wyn/VLAA/PromptVLA
python -m lerobot.scripts.export_pi05_prompt_init check \
  /path/to/prompt-inits/pi05-base-vlm2-init-seed1.safetensors --tokens 2

env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
RUN_PREFIX=pi05-init1-train0 \
PROMPT_INIT_PATH=/path/to/prompt-inits/pi05-base-vlm2-init-seed1.safetensors \
GPU_IDS=0,1 BATCH_SIZE=32 VLM_PROMPT_TOKEN_COUNTS="2" \
RANK_FREQ=250 RANK_SAMPLES=16 RANK_OUTPUT_PROBES=4 \
RANK_SEED=2026 RANK_TIMESTEPS='[0.2,0.5,0.8]' \
bash examples/training/train_pi05_vlm_only_rank_sweep.sh 0 \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

The run is `pi05-init1-train0-base-vlm2-seed0`. Initialization import occurs **after**
normal seed0 model construction and source-weight loading, **before** DDP, optimizer
creation and step-zero diagnostics. It copies values in-place, draws no randomness,
and never restores the source run's RNG or optimizer state. This keeps seed0's data
ordering, augmentation (if enabled), flow noise **and time sampling**, not just its
Gaussian noise. It is an initialization-swap intervention, not a noise-only ablation.
Both DDP ranks verify the same prompt checksum. A shape mismatch, nonfinite values,
or an inexact cast to the target prompt precision is rejected.

The new run retains `prompt_init.safetensors` and `prompt_init.json`, including the
source training seed, source step, replaced/loaded tensor hashes and new training
seed. Normal final checkpoints remain self-contained for evaluation: the external
initialization file is not needed. On trainer resume the import is skipped and the
saved learned weights take precedence, even if the original transfer file is gone.
Full checkpoints are still final-only; rank diagnostics still save tiny snapshots.

For comparison, keep source checkpoint contents, dataset/order, two-GPU world size,
per-device batch, precision, software versions, learning rate and probe settings
fixed. Cross-server copying guarantees the same prompt values; a seed alone cannot
guarantee bitwise identical training on different CUDA/PyTorch/hardware stacks.
Use the current seed0 recheck as control. If the hybrid recovers, it supports an
effect of the remaining training randomness conditional on this initialization;
it does not identify Gaussian noise as the sole cause. If it remains poor, add the
reverse swap (init0/train1) before attributing all failures to initialization.

### Regenerate a missing step-zero prompt without training

If both the original and `pi05-rank-recheck-*` step-zero snapshots are gone,
`PROMPT_EXPORT_PATH` runs the normal seeded dataset/model construction and source
weight loading, exports the actual initial VLM prompt, then exits. It does **not**
create an optimizer, compute rank diagnostics, run training updates, save a full
checkpoint, or start W&B. The original model and dataset must still be available;
the deleted trained checkpoint and its `prompt_rank/manifest.json` are not needed.

For token=8, seed0, using the same two-GPU setup as the training controls:

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_REFERENCE \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
RUN_PREFIX=pi05-init-rebuild \
PROMPT_EXPORT_PATH=/path/to/prompt-inits/pi05-base-vlm8-init-seed0.safetensors \
GPU_IDS=0,1 BATCH_SIZE=32 VLM_PROMPT_TOKEN_COUNTS="8" \
bash examples/training/train_pi05_vlm_only_rank_sweep.sh 0 --wandb.enable=false
```

Only one source/token count is allowed per export. A pre-existing destination is
never overwritten. Both ranks check identical prompt hashes; rank0 writes the
portable file. Success logs `PROMPT_EXPORT ... seed=0 tokens=8 ... 0 optimizer updates`.
The tensor payload is 64 KiB (8 × 2048 × float32), plus provenance metadata.
The metadata labels this as `regenerated_training_initialization` and includes the
source path, current configuration, versions, precision, and actual tensor SHA256.
Although the ordinary launch configuration still says `steps=3000`, export mode
returns before the first optimizer update. Normal training is unchanged when the
export variables are unset.

If the completed `init1/train0` token=8 run still has `prompt_init.json`, add this
environment assignment to the export command (after the `env -u` options):

```bash
PROMPT_EXPORT_REFERENCE=/data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-init1-train0-base-vlm8-seed0/prompt_init.json
```

This compares the regenerated tensor against `replaced_prompt_sha256`: the native
seed0 prompt **before** that run imported seed1. It deliberately does not compare
against `loaded_prompt_sha256`, which belongs to the imported seed1 prompt.
Wrong training seed/token count or a hash mismatch stops the export. A matching
hash is recorded as `historical_hash_verified=true`. Without this reference the
new tensor is valid, but matching a seed number alone does not prove it equals the
deleted historical initialization. Keep the original source weights, model
configuration, software versions and precision to maximize reproducibility.

Then train the missing init0/train1 experiment with the exported file:

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
RUN_PREFIX=pi05-init0-train1 \
PROMPT_INIT_PATH=/path/to/prompt-inits/pi05-base-vlm8-init-seed0.safetensors \
GPU_IDS=0,1 BATCH_SIZE=32 VLM_PROMPT_TOKEN_COUNTS="8" \
RANK_FREQ=250 RANK_SAMPLES=16 RANK_OUTPUT_PROBES=4 \
RANK_SEED=2026 RANK_TIMESTEPS='[0.2,0.5,0.8]' \
bash examples/training/train_pi05_vlm_only_rank_sweep.sh 1 \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

This second command really trains 3000 steps and saves only the final full model;
the first command only regenerates its initialization. Training seed1 still
controls noise, time sampling, data order and other training randomness together.

## What is being measured

For fixed observations `x_i`, Gaussian noise `epsilon_i`, and normalized recorded
actions `a_i`, form `z_it = t*epsilon_i + (1-t)*a_i`. Let `J_it` be the Jacobian of
the fully observation-conditioned flow **velocity** with respect to all learned
VLM prompt coordinates. Keep only real action channels and non-padded timesteps.

For `R` fixed, shared Rademacher output vectors `u_r`, stack

`G[(i,t,r),:] = u_r^T J_it / sqrt(R)`.

Compute `K = G G^T` on CPU in float64. There is **no parameter-axis sketch**.
Let `lambda` be the nonnegative eigenvalues of K and `sigma = sqrt(lambda)`:

| Metric | Definition |
|---|---|
| `jacobian_erank` | `exp(-sum(p*log(p)))`, `p = sigma/sum(sigma)` |
| `ntk_erank` | Same entropy, but `p = lambda/sum(lambda)` |
| `participation_rank` | `(sum(lambda))^2 / sum(lambda^2)` |
| `stable_rank` | `sum(lambda)/max(lambda)` |
| `rank90_energy`, `rank95_energy` | Minimum leading components covering 90%/95% of squared singular-value energy |
| `tangent_trace` | `sum(lambda)`; projected total tangent energy, not normalized by sample count |
| `energy_per_parameter` | `tangent_trace / trainable_parameter_count` |
| `rank_ceiling` | `min(number_of_projected_rows, trainable_parameter_count)` |

The zero Jacobian has zero ranks and energy. Near-null eigenvalues at float64
roundoff are discarded before taking square roots. The predictor retains training
precision (normally bf16 with accelerate autocast); this is not a full-FP32 study.

**Interpretation:** these are spectra of a fixed **output-projected** Jacobian,
not the exact full-model Jacobian, loss-gradient covariance, prompt embedding rank,
attention rank, or the Jacobian through full denoising/closed-loop rollouts.
A one-token prompt has 2048 coordinates; its measured Jacobian rank can exceed 1.
The current diagnostic does not establish whether target action corrections align
with the measured tangent subspace.

## Fixed protocol and controls

Defaults: step **0,250,500,...,3000** (13 measurements), 16 observations,
times `[0.2,0.5,0.8]`, 4 output probes, probe seed 2026. The row ceiling is **192**.
Observation indices are sampled once, one random frame per dataset stratum;
this is not guaranteed to cover all 40 tasks. Fewer dataset rows reduce the ceiling.

The probe seed is independent of training seed and prompt count. All six runs use
the same observations, disabled image augmentation, noise, timesteps and projections.
Processed observations are cached once on CPU, with sample identities and a hash.
Training processors and RNG streams are preserved. For cross-run comparison, verify
`sample_sha256`, action dimensions, time settings, precision and source/normalization
in the manifests. A raw base checkpoint with different processors is not a matched
zero-token performance control.

Each rank probes a subset of samples. A temporary prompt leaf is used outside the
DDP reducer and optimizer; the original parameter object, gradients and individual
module training flags are restored even after a Python exception. Samples are
gathered in a world-size-independent order. Ranks synchronize after each sample/time
forward and its VJPs, rather than waiting for one long rank-0-only analysis.
FSDP/DeepSpeed, compile, LoRA and non-VLM-only training are rejected for this diagnostic.

The monitor records time cost. If it is too slow, increase `RANK_FREQ=500` first.
Changing sample/projection counts changes the estimator, so use the same settings
throughout a comparison. If ranks approach the ceiling, confirm on a larger common
probe (e.g. `RANK_OUTPUT_PROBES=8`) before interpreting saturation as model capacity.

## Outputs

Within each model's training output directory:

```text
prompt_rank/
  manifest.json       # definitions, settings, sample identities, hash, provenance
  fixed_probe.pt      # processed observations, fixed Gaussian noise/output probes
  metrics.csv         # one row per measured completed step
  metrics.jsonl       # the same scalar records
  step_000000.npz     # small prompt-only snapshot + Gram/eigenvalues/singular values
  ...
  step_003000.npz
checkpoints/003000/pretrained_model/
```

W&B metrics are under `eval/prompt_rank/`, especially `jacobian_erank`, `ntk_erank`,
`rank95_energy`, `rank_ceiling` and `probe_seconds`, plotted against training step (`_step`).
The local CSV/JSONL remains available if W&B is disabled.

For an individual run, pass `--policy.compile_model=false --prompt_rank.freq=250`
and optional `--prompt_rank.samples`, `--prompt_rank.output_probes`,
`--prompt_rank.timesteps`, `--prompt_rank.seed` to `train_pi05_vlm_only.sh`.
The general token sweep accepts the same flags; monitoring defaults to disabled
outside the dedicated rank sweep.

## 训练完成后的评估与分析（seed1）

这部分复用已有模型、`fixed_probe.pt` 和 prompt 快照，不需要重新训练。
默认分析 base 的 1/2/4/8/16/32 token，训练 seed1；LIBERO 评估 seed 仍为 1000，
每个任务 10 episodes、每个模型共 400 episodes。训练 seed 与评估 seed 不要混淆。

先仅重画已有数据（CPU，不加载模型）：

```bash
bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh plots 1
```

完整流程：六个模型依次评估 → 离线误差对齐 → 汇总和绘图。

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u BASE_CKPT -u BASE_MODEL_PATH \
CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
BASE_OUTPUT=/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval \
REPORT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis \
VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" \
GPU_ID=1 \
bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh all 1
```

可以将 `all` 分别换成 `eval`、`align`、`report` 分阶段执行。`report` 要求所有所选模型
已有完整评估和对齐记录，缺失时会报错。`plots` 只使用训练时保存的谱数据。
评估复用现有的单行进度条和按任务续测；对齐按模型/step 续算。模型或分析协议变化时，
选择新的输出位置，而不是混用旧结果。设置 `DRY_RUN=true` 可只检查调度命令。

对齐默认在单张 GPU 上分析 step **0、1500、3000**。每个模型只加载一次最终 checkpoint，
再恢复各 step 的小型 prompt 快照。该流程要求 backbone 一直冻结；会检查 VLM-only 配置、
训练 seed、checkpoint step、最终 prompt 和缓存哈希。重算的 Gram 与原记录的相对 Frobenius
差异默认不得超过 2%，用于发现加载、精度或探针不一致；超过阈值应排查原因。
观测和语言 token 已保存在缓存中，因此不再读取数据集或加载 tokenizer。

### 误差对齐的定义与解释

在原来的固定观测、噪声和时间点上，flow matching 的目标为 `u = noise - action`，
当前残差为 `e = u - velocity`。保留原输出投影 `S`，得到 `G = S J` 和 `b = S e`。
若 `G G^T` 的前 k 个正特征值对应向量为 `U_k`，则：

`top_k_alignment = ||U_k^T b||^2 / ||b||^2`。

它测量**投影残差**与主导切空间的重合程度。图中的 `k/M` 是各向同性方向的参考值，
不是经统计检验得到的随机基线。满投影空间可能几乎总能覆盖残差，因此不要只看全秩覆盖率。
`alignment_at_rank95` 则使用覆盖 95% 切向能量的方向数；有效秩与这个方向数不是同一个量。
零残差时对齐比例无定义，保存为 null；不够 k 个正特征方向时对应 top-k 也保存为 null。

另生成独立 Rademacher 输出投影 `T`（默认 4 个，seed2027），在**同一批观测、噪声和时间点**
上得到 `H = T J`、`c = T e`。在原投影中拟合 ridge 线性修正：

`alpha = (G G^T + lambda I)^-1 b`, `delta = G^T alpha`,
`lambda = RIDGE_RELATIVE * trace(G G^T) / M`，默认 `RIDGE_RELATIVE=0.01`。

记录 `heldout_linear_reduction = 1 - ||c - H delta||^2 / ||c||^2`：

- 接近 1：该线性修正在独立输出投影上也能明显降低残差。
- 接近 0：改善有限。
- 小于 0：在独立投影上反而增加误差，原投影中的拟合可能过于乐观。

**没有把 delta 写入模型，也没有做额外训练。** 这仍是局部线性预测，不等于实际非线性更新后的
loss 或 rollout 成功率；`linear_delta_relative_norm` 很大时尤其要谨慎。
这里的 held-out 仅指新的输出投影，不能声称新观测、新任务或闭环控制的泛化。
它补充了谱几何证据，尚不构成 prompt 提升成功率的因果证明。

对齐输出保存于每个 run 的 `prompt_alignment/`，包括协议、每个 step 的 JSON 指标和小型 NPZ。
NPZ 保存训练/独立投影的 Gram、交叉 Gram 及残差，可离线调整 ridge 分析；不保存完整 Jacobian。
更改 `HELDOUT_PROBES`、`HELDOUT_SEED` 或 `RIDGE_RELATIVE` 时，需对所有 token 使用相同设置，
并通过 `ALIGNMENT_NAME` 选择新目录。

### 报告内容

默认完整报告目录：
`/data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/pi05-base-seed1/complete/`。
仅画谱的 `plots` 模式使用同级 `spectra/` 子目录。

| 文件 | 内容 |
|---|---|
| `rank_dynamics.png/pdf` | Jacobian/NTK 有效秩、95% 能量方向数、总切向能量随训练变化 |
| `spectra_and_energy.png/pdf` | 初始/中期/最终的归一化奇异值谱和累计平方能量 |
| `residual_alignment.png/pdf` | 最终 top-k 残差覆盖与独立投影线性改善随训练变化 |
| `success_vs_geometry.png/pdf` | 最终成功率与几何/对齐指标的散点图，数字标记 token 数 |
| `final_summary.csv` | 最终指标、初始指标及其变化、四个 suite/overall 的成功率与分子分母 |
| `history.csv`, `spectra.csv` | 全部训练曲线与作图谱数据 |
| `alignment_summary.csv` | 所选 step 的误差对齐结果 |
| `correlations.csv` | 各 suite/overall 与指标的 Pearson/Spearman 描述性相关，常量指标返回空值 |
| `report.json` | 协议、汇总结果与解释限制 |

比较前会校验固定探针协议、模型来源/归一化与精度；评估必须包含完整的 40 个任务，且
checkpoint、评估 seed、episodes、action chunk 执行长度及评估代码版本匹配。
当前六个点只有一个训练 seed，相关性不代表因果，不能据此选择普遍最优 token 数。
后续建议补多个训练 seed、扩大共同探针，并用独立观测验证误差修正方向。
默认字体为 DejaVu Serif；设置 `FONT_PATH=/absolute/path/times.ttf` 可使用 Times New Roman。
