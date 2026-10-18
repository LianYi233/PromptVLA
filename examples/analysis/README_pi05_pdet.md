# PI05 base、token=2 的 P / D / E / T 实验

目标：在同一台服务器上检查性能退化是否出现，再定位哪些随机因素的替换能够恢复性能。
冻结 backbone 和 action expert，仅训练 2 个 VLM prompt token（4096 个参数）。

| 因素 | 含义 | 实现 |
| --- | --- | --- |
| P | prompt 初始张量 | 导入保存的 seed0 / seed1 初始化，校验元数据和 SHA256，不重新随机生成 |
| D | 训练数据顺序 | EpisodeAwareSampler 使用独立 data_seed；原来的按 epoch 洗牌逻辑不变 |
| E | flow matching 的 Gaussian 动作噪声 ε | 独立 torch.Generator，由 E、训练 step、DDP rank 决定 |
| T | flow matching 的 Beta 时间 t | 沿用模型 sample_time 的分布/缩放，在隔离的 CPU RNG 上采样 |

其他随机性固定为全局 seed=0，数据增强关闭。E/T 绑定每一步的 batch 位置，
不绑定样本 ID；替换 D 时，同一组噪声/时间作用到不同观测上。两个 rank 使用不同 E/T 序列。
有效秩探针独立使用 RANK_SEED=2026，不能把它解读为训练种子。

**这是新的独立随机流协议，不是历史全局 seed1 轨迹的逐步重放。**
即使全 1 组不退化，也只能说明新协议下没有复现，不能排除历史随机流的耦合、服务器环境等因素。
P1 不保证坏，P0 不保证好；两者是初始化身份标签，实际张量以哈希为准。

## 七组训练

| PDET_CASES | 模型名后缀 | 对照意义 |
| --- | --- | --- |
| 1111 | P1-D1-E1-T1 | 全 1 参考组，先看是否出现退化 |
| 0000 | P0-D0-E0-T0 | 全 0 参考组 |
| 0111 | P0-D1-E1-T1 | 相对 1111 只替换 P |
| 1011 | P1-D0-E1-T1 | 只替换 D |
| 1101 | P1-D1-E0-T1 | 只替换 E |
| 1110 | P1-D1-E1-T0 | 只替换 T |
| 1000 | P1-D0-E0-T0 | 保留 P1，替换全部 D/E/T；可与 0000 比较 P 的条件效应 |

默认完整模型名例如 `pi05-pdet-base-vlm2-P1-D1-E1-T1`。
每组独立从同一个 base 开始，依次使用两张卡；每卡 batch 32，总 batch 64。
训练 3000 steps，仅保存最终完整 checkpoint `003000`，关闭 torch.compile。
同时保留 step 0、每 250 步和最终步的小型 prompt/有效秩/NTK 记录。

本轮在本地 A100 服务器重新生成 P0/P1，作为一对新的初始化；不假定它们与旧服务器相同。
先在已配置的 lerobot 环境执行（仅导出 token=2 初值，零训练更新）：

```bash
cd /home/wyn/VLAA/PromptVLA
bash examples/setup/prepare_promptvla_local.sh assets
GPU_IDS=0,1 BATCH_SIZE=32 bash examples/setup/prepare_promptvla_local.sh inits
```

资源下载到 `/data1/datasets/libero-assets` 并配置到当前 LIBERO 安装。
初始化导出需要加载完整模型，但不创建优化器、探针或完整 checkpoint，不启动 W&B。
已有初始化文件不会覆盖；若只有 seed0 导出成功，使用 `INIT_SEEDS=1` 补齐 seed1。
确认以下两个文件存在（不需要旧服务器的完整训练模型）：

```text
/data1/wyn/prompt-work/pi05-base-vlm2-init-seed0.safetensors
/data1/wyn/prompt-work/pi05-base-vlm2-init-seed1.safetensors
```

之后在同一环境运行本地 P/D/E/T 训练：

```bash
cd /home/wyn/VLAA/PromptVLA
git pull --ff-only origin main

env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
PROMPT_INIT_DIR=/data1/wyn/prompt-work \
RUN_PREFIX=pi05-pdet GPU_IDS=0,1 BATCH_SIZE=32 \
PDET_CASES="1111 0000 0111 1011 1101 1110 1000" \
RANK_FREQ=250 RANK_SAMPLES=16 RANK_OUTPUT_PROBES=4 \
RANK_SEED=2026 RANK_TIMESTEPS='[0.2,0.5,0.8]' \
bash examples/training/train_pi05_vlm_only_pdet_sweep.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

若想分阶段，先设置 `PDET_CASES="1111 0000"` 跑两个参考组，再用
`PDET_CASES="0111 1011 1101 1110 1000"` 补齐。`DRY_RUN=true` 只检查并打印命令。
已有目录不会覆盖或自动跳过，中断后明确选择剩余 case；新一轮可改 RUN_PREFIX。
输入模型/tokenizer 默认在 `/home/wyn/models/`，数据集在 `/home/wyn/datasets/libero`。
模型输出默认在 `/data1/wyn/chkpt/2601-lerobot/PromptVLA`，日志在 `/data1/wyn/chkpt/2601-lerobot/logs`。
`PROMPT_INIT_DIR` 默认 `/data1/wyn/prompt-work`；七组实验统一使用准备步骤导出的同一对初值。

## 检查隔离是否有效

每个 run 都保存 `prompt_init.safetensors` 和 `prompt_init.json`。
`pdet/manifest-rank*.json` 记录因素、实际 P 哈希、软件版本、Git commit、GPU、batch 和训练配置。
`pdet/inputs-rank*.jsonl` 保存每个 rank 前 8 步的 sample IDs、noise/time 的 SHA256 与 shape。
这些文件很小，应随结果一起保留。审计记录仅覆盖最初 8 步，不证明整段训练数值完全确定。

七组完成后（或至少两组走过前 8 步后），比较输入：

```bash
python -m lerobot.scripts.pdet_experiment audit \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-pdet-base-vlm2-P*-D*-E*-T*
```

例如 1111 与 1101 必须 P、sample IDs 和 time 完全相同，只有 noise 不同。
若检查未通过，先排查隔离/配置，不用成功率解释因素作用。
只有一个初值/随机序列对时，结论是这对 seed 的条件效应，不是所有 seed 的平均因果效应。
若替换单个因素都无效、替换 D/E/T 全部后有效，后续补二因素组合检查交互。
若多个单因素替换均能恢复，说明退化可能依赖组合，不能唯一归因给其中一个因素。

审计会分别报告 `random_inputs` 和 `code_revision`。记录的 Git 提交不同时，
自动在本地比较两个提交的文件差异；仅文档、测试、独立 LIBERO 评估入口或审计入口变化时，
继续审计并报告 `non_training_changes_only`，同时列出实际提交和变更文件。
例如 `f3809e18` 到 `522f1340` 只新增了评估入口并更新测试/文档，训练代码相同。
训练源码、启动脚本、依赖或未分类文件变化仍会报错；不会改写已有 manifest。
本地缺少记录的提交时应先获取相应 Git 历史。
诊断时可加 `audit --inputs-only ...`，但此时只能得到 `inputs_only_passed`，
代码版本标记为 `not_verified`，不能当作完整审计通过。历史未提交改动没有记录，
此检查也不能证明它们不存在。

七组完成后，补齐初始化/数据对照及最差组的噪声/时间替换：

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
PROMPT_INIT_DIR=/data1/wyn/prompt-work \
RUN_PREFIX=pi05-pdet GPU_IDS=0,1 BATCH_SIZE=32 \
PDET_CASES="0011 1001 1010" \
RANK_FREQ=250 RANK_SAMPLES=16 RANK_OUTPUT_PROBES=4 \
RANK_SEED=2026 RANK_TIMESTEPS='[0.2,0.5,0.8]' \
bash examples/training/train_pi05_vlm_only_pdet_sweep.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

沿用已保存的同一对 P0/P1 初值、相同数据和训练配置；每组仍为两卡、总 batch 64、
3000 steps、只保存最终完整模型。已有的七组目录不会被本命令选中。
补充评估时用下面评估命令并将 `PDET_CASES` 同样改为 `"0011 1001 1010"`。

七组按训练顺序依次测试（前五组先测，补训的 `1110`、`1000` 最后测）：

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u BASE_CKPT -u BASE_MODEL_PATH \
CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
BASE_OUTPUT=/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
RUN_PREFIX=pi05-pdet GPU_ID=2 CHECKPOINT_STEP=3000 \
EVAL_SEED=1000 EPISODES_PER_TASK=10 PROGRESS=true \
PDET_CASES="1111 0000 0111 1011 1101 1110 1000" \
WAIT_FOR_CHECKPOINTS=true \
bash run_eval_libero_vlm_only_pdet_sweep.sh all
```

默认使用 2 号 GPU，训练继续使用 0、1 号 GPU；每个模型测试四个 suite，共 400 个 episode。
终端只显示一条实时进度及每个模型最终成功率，详细输出和视频路径保存在 `LOG_ROOT/eval`。
按顺序逐个检查模型，后面的模型未训练完不会阻止前面的模型评估。
轮到未保存完的模型时，等待训练器在完整保存后发布的 `checkpoints/last` 指向 `003000`；
仅有目录或正在写入的 `model.safetensors` 不算完成。默认每 30 秒检查，每个模型最多等 6 小时
（`WAIT_INTERVAL_SECONDS` / `WAIT_TIMEOUT_SECONDS` 可调整）；训练失败时不会自动重启训练。
评估前核对保存的训练配置中 P/D/E/T 是否与目录名一致，并沿用原 VLM-only 模型/处理器检查。
评估失败即停止并显示日志路径；重新执行时沿用逐任务续测检查。
只补测最后两组可设置 `PDET_CASES="1110 1000"`。

单模型评估也可直接使用既有脚本，固定 EVAL_SEED，与训练 D/E/T 分开：

```bash
PRETRAINED_PATH=/home/wyn/models/lerobot/pi05_libero_base \
TOKENIZER_PATH=/home/wyn/models/paligemma-3b-pt-224 \
CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
RUN_NAME=pi05-pdet-base-vlm2-P1-D1-E1-T1 \
BASE_OUTPUT=/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval \
GPU_ID=1 VLM_PROMPT_TOKENS=2 CHECKPOINT_STEP=3000 EVAL_SEED=1000 \
bash run_eval_libero_vlm_only.sh base trained all 0
```

同一套任务初态和评估噪声下比较逐 episode 的成功/失败。400 个评估 episode 不等于
400 次独立训练重复。若全 1 组不退化，下一轮优先增加训练随机流种子或在新机器上另跑
原全局 seed1 协议对照，不要把新协议的结果当作已经推翻历史异常。

恢复训练时必须保留同一 run 的 `pdet/manifest-rank*.json`，保持 P/D/E/T、全局 seed、
world size、batch size、dtype。E/T 由 step/rank 重建，无需保存额外 generator 状态。
跨硬件/PyTorch 版本不承诺逐位一致；不能在对照组间混用版本、batch 或学习率。
