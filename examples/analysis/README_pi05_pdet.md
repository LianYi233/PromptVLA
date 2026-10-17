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

先确认这些文件都在（不需要旧服务器的完整训练模型）：

```text
/root/autodl-fs/wyn/prompt-inits/pi05-base-vlm2-init-seed0.safetensors
/root/autodl-fs/wyn/prompt-inits/pi05-base-vlm2-init-seed1.safetensors
```

在新 AutoDL 服务器运行：

```bash
cd /root/PromptVLA
git pull --ff-only origin main

env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH -u PROMPT_EXPORT_REFERENCE \
OUTPUT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
LOG_ROOT=/root/autodl-tmp/logs/vlm-only \
PROMPT_INIT_DIR=/root/autodl-fs/wyn/prompt-inits \
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
输入模型、tokenizer 和数据集默认在 `/root/autodl-fs/wyn/`，模型输出默认在 autodl-tmp。

## 检查隔离是否有效

每个 run 都保存 `prompt_init.safetensors` 和 `prompt_init.json`。
`pdet/manifest-rank*.json` 记录因素、实际 P 哈希、软件版本、Git commit、GPU、batch 和训练配置。
`pdet/inputs-rank*.jsonl` 保存每个 rank 前 8 步的 sample IDs、noise/time 的 SHA256 与 shape。
这些文件很小，应随结果一起保留。审计记录仅覆盖最初 8 步，不证明整段训练数值完全确定。

七组完成后（或至少两组走过前 8 步后），比较输入：

```bash
python -m lerobot.scripts.pdet_experiment audit \
  /root/autodl-tmp/chkpt/2601-lerobot/vlm-only/pi05-pdet-base-vlm2-P*-D*-E*-T*
```

例如 1111 与 1101 必须 P、sample IDs 和 time 完全相同，只有 noise 不同。
若检查未通过，先排查隔离/配置，不用成功率解释因素作用。
只有一个初值/随机序列对时，结论是这对 seed 的条件效应，不是所有 seed 的平均因果效应。
若替换单个因素都无效、替换 D/E/T 全部后有效，后续补二因素组合检查交互。
若多个单因素替换均能恢复，说明退化可能依赖组合，不能唯一归因给其中一个因素。

单模型评估可直接使用既有脚本，固定 EVAL_SEED，与训练 D/E/T 分开：

```bash
PRETRAINED_PATH=/root/autodl-fs/wyn/models/pi05_libero_base \
TOKENIZER_PATH=/root/autodl-fs/wyn/models/google/paligemma-3b-pt-224 \
CKPT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
RUN_NAME=pi05-pdet-base-vlm2-P1-D1-E1-T1 \
BASE_OUTPUT=/root/autodl-tmp/eval/vlm-only \
GPU_ID=1 VLM_PROMPT_TOKENS=2 CHECKPOINT_STEP=3000 EVAL_SEED=1000 \
bash run_eval_libero_vlm_only.sh base trained all 0
```

同一套任务初态和评估噪声下比较逐 episode 的成功/失败。400 个评估 episode 不等于
400 次独立训练重复。若全 1 组不退化，下一轮优先增加训练随机流种子或在新机器上另跑
原全局 seed1 协议对照，不要把新协议的结果当作已经推翻历史异常。

恢复训练时必须保留同一 run 的 `pdet/manifest-rank*.json`，保持 P/D/E/T、全局 seed、
world size、batch size、dtype。E/T 由 step/rank 重建，无需保存额外 generator 状态。
跨硬件/PyTorch 版本不承诺逐位一致；不能在对照组间混用版本、batch 或学习率。
