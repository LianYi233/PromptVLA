# 从官方 LIBERO v044 模型继续训练双 prompt

目标：判断已经在 LIBERO 上微调过的 `lerobot/pi05_libero_finetuned_v044`，
再冻结 backbone、只训练 VLM/action prompt 后，成功率是否有**净提升**。
这组实验独立于之前从 `pi05_libero_base` 出发的 token sweep。

| 对照 | 模型 | 用途 |
| --- | --- | --- |
| baseline | 官方 v044，不插入两组 prompt | 本地评测下的原模型基线 |
| initial | 同一个训练 run 的 `000000`，两组随机 prompt，尚未更新 | 测量插入 prompt 本身的影响 |
| trained | 同一 run 的 `003000` | 测量训练后的净效果 |

默认 16 个 VLM token + 16 个 action token，两个 embedding 宽度分别为 2048/1024，
共训练 49,152 个参数。VLM、视觉编码器、action expert、action 输入/输出投影均冻结。
使用 flow matching，CABO、next-action priming、bridge、LoRA 均关闭；batch size=32，
3000 次更新，训练 seed=0。学习率和调度器继承源模型配置（峰值 2.5e-5）。
训练日志现有的 `Trainable parameter report` 应只有两个 prompt 权重，
`action_projection` 和 `other` 均为 0。

官方配置使用 bfloat16、MEAN_STD 和保存的归一化统计量。
新入口用 `--policy.path` 加载完整配置，并通过
`--preserve_pretrained_normalization=true` 保留源统计量、特征和归一化方式；
旧训练入口仍保留原有行为。不要只替换旧脚本的 `PRETRAINED_PATH`。
模型权重、预处理、后处理和两个 normalization safetensors 文件必须完整下载。

```bash
git switch prompt-learning
git pull --ff-only origin prompt-learning

# 已完整下载则跳过；使用已安装本仓库的训练环境。
hf download lerobot/pi05_libero_finetuned_v044 \
  --local-dir /root/autodl-tmp/models/pi05_libero_finetuned_v044

# 建议先确认原模型在本地能正常工作；不需要等待 prompt 训练。
GPU_ID=0 bash run_eval_libero_finetuned_prompt.sh baseline all 0

# 默认开启 W&B：prompt-learning / online / disable_artifact=true。
GPU_IDS=0 BATCH_SIZE=32 \
  bash examples/training/train_pi05_finetuned_prompt.sh 0

# 完成后评测原模型、确切的初始化模型和最终模型；已完成的基线可复用原评测缓存。
GPU_ID=0 bash run_eval_libero_finetuned_prompt.sh all all 0
```

默认模型目录：
`/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/finetuned-v044/pi05-ftv044-dual-vlm16-act16-seed0/`。
保存完整模型 `000000`、`001000`、`002000`、`003000`，需要为四份模型预留空间。
设置 `SAVE_FREQ=3000` 可只保留初始化和最终模型。
日志在 `/root/autodl-tmp/logs/prompt-learning/finetuned-v044`；
评测在 `/root/autodl-tmp/eval/prompt-learning/finetuned-v044`。
`DRY_RUN=true` 检查路径、模型和统计量并打印命令，不启动训练或仿真。

```bash
# 观察训练途中是否先改善后退化；事先确定主结果用最终 step，避免挑最高测试点。
CHECKPOINT_STEP=1000 GPU_ID=0 bash run_eval_libero_finetuned_prompt.sh trained all 0
CHECKPOINT_STEP=2000 GPU_ID=0 bash run_eval_libero_finetuned_prompt.sh trained all 0

# 可选：增加训练 seed，使用不同目录和同一组评测初始状态。
GPU_IDS=0 bash examples/training/train_pi05_finetuned_prompt.sh 1
GPU_ID=0 bash run_eval_libero_finetuned_prompt.sh all all 1
```

三组评测均为四套 LIBERO、每任务 10 次、环境初始 seed=1000、每次执行 10 个动作，
每模型共 400 episodes。`EVAL_SEED`、`EPISODES_PER_TASK` 可统一调整。
原模型会显式设 VLM/action prompt 数为 0；其余模型加载各自真实保存的 prompt，
不会在评测时重新初始化或改变长度。评测前校验源/候选模型的架构、dtype、prompt 数，
以及归一化配置和统计量数值是否一致。种子相同不代表不同结构的策略使用相同的去噪噪声流。

记成功率为 S_base、S_init、S_train：

- **S_train − S_base** 是本实验的主要结论：对已微调模型是否有净增益。
- S_init − S_base 是 prompt 插入与初始化带来的变化；随机甚至全零 prompt 都不保证函数不变。
- S_train − S_init 衡量学习相对于随机 prompt 的改善。
- 若 S_train > S_init 但 S_train < S_base，说明训练改善了插入后的模型，但未超过原模型。
- 若本地 v044 基线仍接近 0%，先排查 checkpoint/数据/图像与动作处理兼容性，暂不归因于 prompt。

报告每套 suite、每任务的成功率和训练 seed；每 suite 100 episodes 时，1 个 episode 就是 1 个百分点。
几百分点差异需更多训练 seed 和 episodes 才能判断稳定性。原模型已接受 LIBERO 训练，
该实验检验继续适配，不足以证明未见任务的泛化能力。结果可能提升、持平或下降，事前不作性能承诺。

源配置：
https://huggingface.co/lerobot/pi05_libero_finetuned_v044/blob/main/config.json
