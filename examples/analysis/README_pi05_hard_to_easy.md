# 固定 LIBERO 起点，只改变难度顺序

## 可选：VLM prompt + 完整 action expert

默认旧版仍只训练 VLM prompt。显式设置 `TRAIN_ACTION_EXPERT=true` 才解冻完整动作生成网络：
`gemma_expert.model` 的全部 Transformer 层及 AdaRMS、`action_in_proj`、`action_out_proj`、
`time_mlp_in`、`time_mlp_out`，同时训练2个VLM prompt token。视觉编码器、VLM主干仍冻结；
action prompt=0。Gemma包装器里不参与动作预测的词表 `lm_head` 不训练。
`train_action_projections` 保持false，因为它是另一个“仅投影层”开关；完整范围由
`policy.train_action_expert=true` 指定，`policy.vlm_prompt_only=false`。

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
OUTPUT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA \
LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs \
SOURCE_PAIR_DIR=/data1/wyn/prompt-work/source-pair-r1 \
SCORES_PATH=/data1/wyn/prompt-work/frameskip-r1/scores.npz \
ORDER_DIR=/data1/wyn/prompt-work/hard2easy-r1 \
TRAIN_ACTION_EXPERT=true \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29583 REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_hard_to_easy.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

新模型名 `pi05-hard2easy-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1`，不覆盖prompt-only模型。
起点仍为 `source-pair-r1/libero` 中的 `pi05_libero_base` 权重；复用相同H2E计划、P1、E1/T1、
Bocc、2卡×batch32、3000步、bf16、无compile、相同学习率/损失，仅保存完整003000。
训练启动打印实际可训练参数分组；模型和优化器会拒绝意外解冻VLM或漏训动作模块。
完整动作网络的梯度/优化器状态比prompt-only占用更多显存；未在本开发环境验证A100显存峰值。

有效秩仍只测输出对VLM prompt的Jacobian，并不表示整个动作网络的有效秩。
探针期间临时冻结动作参数以隔离DDP/优化器梯度，结束或异常后恢复全部参数、梯度引用和训练模式。
训练结束的审计显式记录两项干预：难到易顺序、完整action expert训练。因此与旧67.75%结果的
差异不能单独归因于课程；若要隔离课程效应，还需要同样解冻范围的随机顺序对照。

测试新模型时同样指定解冻模式（用于选择和检查checkpoint，不会在评估中训练）：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
TRAIN_ACTION_EXPERT=true GPU_ID=0 REPEAT_ID=1 \
EVAL_SEED=1000 EPISODES_PER_TASK=10 PROGRESS=true DRY_RUN=false \
bash run_eval_libero_hard_to_easy.sh all
```

下文描述默认 `TRAIN_ACTION_EXPERT=false` 的prompt-only对照和两种模式共用的课程定义。

直接对照为 `pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1`（本轮 overall 67.75%）。
继续使用 `pi05_libero_base`，不训练 generic。共用原 source-pair 的 LIBERO 权重视图、P1 初始化、
处理器和数据；VLM token=2、action prompt=0、全局seed0、D0、E1/T1、Bocc，2 GPU × 每卡batch32，
3000步、bf16、compile=false，只保存完整003000；学习率、原始未加权损失、rank/flow诊断均不变。

## “难”是什么

复用 `/data1/wyn/prompt-work/frameskip-r1/scores.npz` 的 `score`。不重新跑视觉评分。
旧默认评分为0.6×动作变化分数 + 0.2×视觉变化分数 + 0.2×夹爪事件邻域指示值；实际配置以
缓存的 `score_recipe` 为准。动作项含相邻动作变化及短期未来方差，视觉项含视觉/动作变化比值。
动作/视觉分量在episode内做稳健归一化。它是**难度/关键性的代理分数**，不是当前模型loss，
也不是经过验证的跨episode绝对难度。高分先训练不必然更好，本次正是检验这个安排。

## 顺序如何变化

从原始 `source-pair-r1/D0-W0.npz` 读取192,000个样本使用实例（3000×64），不重新抽样。
每个全局位置的task ID保持原样，只把同一task的使用实例按分数从高到低放回这些位置。
相同分数保持原W0中的相对顺序。每个实例的occurrence ID不变，E1/T1随实例移动。
同一帧的多次使用仍是不同实例，不会被合并；它们仍保留各自原有噪声/时间。

因此这是**task内部的frame/chunk-start课程**，不是先把某个task全部学完，也不打乱样本内部
连续50步动作目标。每个task的难度单调下降，但不同task在batch内混合，故全局batch均值不必逐步单调。
没有新增500步数据warmup、筛样、重复采样或关键帧loss加权；原学习率warmup保持不变。
它不是把旧FrameSkip C采样比例调度倒转，也不是Kearly/Klate的loss权重调度。

## 训练一次

仓库根目录、已激活训练环境后执行：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
SOURCE_PAIR_DIR=/data1/wyn/prompt-work/source-pair-r1 \
SCORES_PATH=/data1/wyn/prompt-work/frameskip-r1/scores.npz \
ORDER_DIR=/data1/wyn/prompt-work/hard2easy-r1 \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29583 REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_hard_to_easy.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

自动在CPU上准备并验证 `H2E.npz`、`curriculum.json`、逐step的 `schedule.csv`，已有完全匹配的
计划可复用。需保留 source-pair、原模型、评分缓存和参考运行。缺少缓存则报错，不悄悄重算/替换分数。
`DRY_RUN=true`只准备小文件并打印命令。不会覆盖已有训练目录。

模型名：`pi05-hard2easy-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1`。
输出默认 `/data1/wyn/chkpt/2601-lerobot/PromptVLA`，日志默认 `/data1/wyn/chkpt/2601-lerobot/logs`。
只启动一个双卡训练，GPU2、3不自动启动其他实验。

## 审计与测试

训练结束自动审计所有3000步的样本/噪声trace，要求原样本使用池、task slots、P1、E1/T1 bank及
训练配方一致，且只改变顺序。跨历史运行的代码哈希单独报告；`inputs_and_recipe_passed`不证明
历史代码/未提交改动完全等价或GPU数值确定。本次新增独立准备/训练/测试入口，不修改原模型、loss、
训练器、采样器及噪声生成代码。

手动重跑审计：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_hard_to_easy audit \
  /data1/wyn/prompt-work/hard2easy-r1 \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-hard2easy-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1
```

测试保持原四套suite、每任务10次、seed1000、单卡、原地进度条：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
GPU_ID=0 REPEAT_ID=1 EVAL_SEED=1000 EPISODES_PER_TASK=10 PROGRESS=true DRY_RUN=false \
bash run_eval_libero_hard_to_easy.sh all
```

比较本次H2E与67.75%直接对照，报告四suite及overall；单次训练不能说明所有种子下的课程效应。
