# 原始 PI05 路径上的 VLM-only prompt 实验

仓库：`LianYi233/PromptVLA`，分支：`main`。分别从 `pi05_libero_base`、`pi05_libero_finetuned_v044` 开始，
只训练 VLM prompt。默认把“一个 prompt”解释为 **1 个可学习 token**；
如果指的是一组 prompt，可设置 `VLM_PROMPT_TOKENS=16` 或其他正整数。

| 设置 | 两个来源共用 |
| --- | --- |
| VLM prompt | 默认 1 × 2048，共 2048 个可训练参数 |
| action prompt | 0，不在 action 序列中插入 token |
| 冻结 | 视觉编码器、VLM、action expert、action 输入/输出投影、时间条件模块 |
| 目标 | 原 observation-conditioned flow matching，3000 次更新 |
| 关闭 | CABO、priming、bridge、LoRA、action projection 训练 |
| 精度 | 默认 bfloat16，可用 `DTYPE=float32`；三组评测跟随该 run 保存的 dtype |
| 优化器 | AdamW，峰值 2.5e-5，weight decay=0.01，不启用额外梯度裁剪 |
| 调度 | 沿用 ablation 的配置：warmup=1000、decay=30000、终值=1e-5；3000 步时调度器自动缩放为 warmup=100、decay=3000 |
| W&B | 开启，project=`prompt-learning`，online，disable_artifact=true |

VLM prompt 位于图像/语言/状态组成的 observation prefix 之后。Observation 保持原来的互相可见关系，
VLM prompt 读取 observations 和自己的 prompt 块，action tokens 读取完整 prefix 与 action 块。
因此梯度经冻结的 action expert、VLM 回到 prompt，冻结不等于对整条路径使用 `no_grad`。
没有新 action head。`vlm_prompt_only=true` 会拒绝启用其他适配路径；优化器再次检查
唯一可训练张量必须是 `model.vlm_prompt_tokens.weight`。

## AutoDL 启动

如果要重新训练 base 的 1/2/4/8/16/32 token、seed1，并记录动作 velocity 对 VLM prompt
的有效秩，使用 `examples/training/train_pi05_vlm_only_rank_sweep.sh 1`。
它每 250 步测量固定探针，仍只保存 3000 步的完整模型；参见
[有效秩监测协议和完整命令](../analysis/README_vlm_prompt_rank.md)。

首次迁移到独立 PromptVLA 目录：

```bash
cd /root
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LianYi233/PromptVLA.git
cd PromptVLA
conda activate lerobot
python -m pip install --no-deps -e .
python -c 'import lerobot; print(lerobot.__file__)'
```

包路径应指向 `/root/PromptVLA/src/lerobot/`。已有模型和输出目录继续使用，无需移动。
这一步会把当前 Python 环境的 editable 安装指向新代码；请等旧目录中的训练结束后再切换。

后续更新：

```bash
cd /root/PromptVLA
git pull --ff-only origin main
```

激活已有 LeRobot 环境并安装此 checkout（`pip install -e .`）。默认路径：

- dataset：`/root/autodl-tmp/datasets/libero`
- base：`/root/autodl-tmp/models/pi05_libero_base`
- v044：`/root/autodl-tmp/models/pi05_libero_finetuned_v044`
- tokenizer：`/root/autodl-tmp/models/google/paligemma-3b-pt-224`

模型目录必须包含完整权重、config 和 pre/postprocessor 文件，v044 还需要它们引用的统计量文件。
缺少模型时可以沿用 `HF_HUB_DISABLE_XET=1 hf download ... --local-dir ... --max-workers 1` 下载。
base 使用本地数据集 `meta/stats.json` 的 state/action `q01/q99`；启动前检查缺失项。
可用 `BASE_PRETRAINED_PATH`、`FINETUNED_PRETRAINED_PATH`、`DATASET_ROOT`、`TOKENIZER_PATH` 指定其他位置。

默认**两组并行，每张卡一个模型，各自 global batch=32**：

```bash
# 先检查路径、模型、统计量、目标目录和两条训练命令
GPU_IDS=0,1 DRY_RUN=true bash examples/training/train_pi05_vlm_only_pair.sh 0

# GPU 0：base；GPU 1：v044
GPU_IDS=0,1 BATCH_SIZE=32 bash examples/training/train_pi05_vlm_only_pair.sh 0
```

如果希望两卡共同训练一个模型，先 base、后 v044，使用 DDP。
`BATCH_SIZE` 始终指每卡 batch；每卡 16 × 2 卡仍为 global batch=32：

```bash
TRAIN_LAYOUT=ddp GPU_IDS=0,1 BATCH_SIZE=16 \
bash examples/training/train_pi05_vlm_only_pair.sh 0
```

两种方式是替代方案，不要对同一输出目录重复启动。并行模式使用不同端口
29531/29532、不同训练目录、日志和编译缓存。任一模型失败不会取消另一个独立训练，
最终退出码会报告失败；只重跑失败来源时使用单模型入口：

```bash
GPU_IDS=1 bash examples/training/train_pi05_vlm_only.sh finetuned 0
```

已有输出会被拒绝覆盖；保留有价值的 checkpoint，选择新 `RUN_NAME` 重跑，
或用训练器的 `--config_path=.../train_config.json --resume=true` 恢复已有 run。

更改 token 数或学习率：

```bash
VLM_PROMPT_TOKENS=16 GPU_IDS=0,1 BATCH_SIZE=32 \
bash examples/training/train_pi05_vlm_only_pair.sh 0 --policy.optimizer_lr=1e-5
```

默认输出根目录：`/root/autodl-tmp/chkpt/2601-lerobot/vlm-only`。
run 名称：`pi05-base-vlm1-seed0`、`pi05-finetuned-vlm1-seed0`。
日志在 `/root/autodl-tmp/logs/vlm-only`。保存 `000000`、`001000`、`002000`、`003000` 的完整模型；
两组共 8 份 checkpoint，需预留相应磁盘空间。`SAVE_FREQ=3000` 可只保存初始化和最终模型。

### 两个来源分别训练 8、16、32、64 个 token，只保存 3000 步

批量入口沿用上面的 VLM-only 配方，默认每次用两张卡训练，按顺序运行八个独立实验：
先 `base-vlm8/16/32/64`，再 `finetuned-vlm8/16/32/64`。
每个实验都从对应的原始模型开始，不从上一个 token 数的模型继续训练。

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
OUTPUT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
LOG_ROOT=/root/autodl-tmp/logs/vlm-only \
GPU_IDS=0,1 BATCH_SIZE=32 VLM_PROMPT_TOKEN_COUNTS="8 16 32 64" \
bash examples/training/train_pi05_vlm_only_token_sweep.sh both 0 \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

此入口默认模型、tokenizer 和数据集从 `/root/autodl-fs/wyn/` 读取，训练输出和日志写到
`/root/autodl-tmp/`。每卡 batch 32、全局 batch 64；学习率保持 `2.5e-5`。
默认关闭 `torch.compile`，保留 bf16 和 gradient checkpointing。
固定 `FLOW_STEPS=3000`、`SAVE_ONLY_FINAL=true`，只保存 `003000`；
token=8/16/32/64 分别训练 16384/32768/65536/131072 个参数。
W&B 默认开启，项目 `prompt-learning`。

输出目录为 `$OUTPUT_ROOT/pi05-{base|finetuned}-vlm{8|16|32|64}-seed0/checkpoints/003000/`，
`last` 只是软链接。每个 checkpoint 仍包含完整冻结模型权重，共保存八份最终模型。
开始前会检查所有选定实验；已有输出不覆盖，也不自动跳过。训练失败后停止后续实验。
可设置 `DRY_RUN=true` 仅检查并打印命令，或选择尚未执行的组合，例如：

```bash
VLM_PROMPT_TOKEN_COUNTS="32 64" GPU_IDS=0,1 BATCH_SIZE=32 \
bash examples/training/train_pi05_vlm_only_token_sweep.sh finetuned 0
```

可用 `RUN_PREFIX=retry` 给失败实验选择新目录。此入口要求取消旧的单模型
`RUN_NAME`、`PRETRAINED_PATH`、`VLM_PROMPT_TOKENS` 环境变量，分别使用
`RUN_PREFIX`、两个来源的路径变量、`VLM_PROMPT_TOKEN_COUNTS` 控制实验。
仍可显式设置 `VLM_PROMPT_TOKEN_COUNTS="2 4"` 选择之前的训练组合。
评测入口不变；例如测 base-token8：

```bash
BASE_PRETRAINED_PATH=/root/autodl-fs/wyn/models/pi05_libero_base \
TOKENIZER_PATH=/root/autodl-fs/wyn/models/google/paligemma-3b-pt-224 \
CKPT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
GPU_ID=0 VLM_PROMPT_TOKENS=8 CHECKPOINT_STEP=3000 \
bash run_eval_libero_vlm_only.sh base trained all 0
```

批量评测这八个模型时，在下面的批量评测命令中设置
`VLM_PROMPT_TOKEN_COUNTS="8 16 32 64"` 即可。

### 依次评测四个 2/4-token 模型，单行进度与最终成功率

批量评测入口默认选择 `base-vlm2`、`base-vlm4`、`finetuned-vlm2`、`finetuned-vlm4`，
依次在同一张 GPU 上评测 `003000`，四套 LIBERO、每套 100 次、每个模型共 400 次。
输入模型/tokenizer 默认在 `/root/autodl-fs/wyn/models/`，checkpoint 默认在
`/root/autodl-tmp/chkpt/2601-lerobot/vlm-only/`，与上面的批量训练目录一致。

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH \
CKPT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
BASE_OUTPUT=/root/autodl-tmp/eval/vlm-only \
GPU_ID=1 VLM_PROMPT_TOKEN_COUNTS="2 4" CHECKPOINT_STEP=3000 \
bash run_eval_libero_vlm_only_token_sweep.sh both all 0
```

默认 `PROGRESS=auto`：在终端显示一条原地更新的进度条，包括第几个模型、当前模型的
已完成/总 episode、耗时和预计剩余时间。每完成一个 episode 更新，不逐帧输出；
模型/环境加载期间显示加载状态，开始 rollout 后才估算 ETA。ETA 针对当前模型，
不同任务的耗时会影响估计；续跑计数包含已保存任务，速度根据本次运行估算。
每个模型完成后显示其名称、四个 suite 和 overall 成功率。
逐任务输出、视频路径和完整异常写入 `BASE_OUTPUT/logs/` 中本次调用独立的日志；
视频仍按原脚本保存。不会从累计 summary 日志读取结果，避免打印上次结果或重复结果。
重定向输出时默认不显示动态进度，避免日志刷屏。`PROGRESS=true` 强制显示，
`PROGRESS=false` 恢复完全静默、只输出最终成功率。
开始前先检查四个 checkpoint；任一评测失败或结果不完整会停止后续模型并提示日志位置。
沿用原评测脚本的逐任务续跑及一致性校验。
已启动的旧进程不会自动切换显示方式；需停止旧进程、更新脚本后用原命令重启。
已完整保存的任务会跳过，未完成的当前任务会重新评测。

`DRY_RUN=true` 只检查并打印命令，不评测或写日志。可选择尚未完成的来源/token 数，
例如 `VLM_PROMPT_TOKEN_COUNTS=4 GPU_ID=1 bash run_eval_libero_vlm_only_token_sweep.sh finetuned all 0`。
与之前一致，默认评测 `EVAL_SEED=1000`；自定义 `EPISODES_PER_TASK` 会相应改变总次数。

### 仅重跑 v044，并只保存最终模型

`SAVE_ONLY_FINAL=true` 同时关闭 `000000` 初始化快照和中间保存，仅在训练结束保存
`003000`；`checkpoints/last` 是指向最终 checkpoint 的软链接，不是另一份模型。
它覆盖 `SAVE_FREQ`；只设置 `SAVE_FREQ=3000` 仍然会保存初始化模型。

下例沿用双卡、每卡 batch 32（总 batch 64）、1 个 VLM token、3000 步和原学习率，
从官方 v044 重新开始。训练输出统一写到 `/root/autodl-tmp/chkpt/2601-lerobot/vlm-only/`，
使用独立 run 名称，并关闭编译。

```bash
export FINETUNED_PRETRAINED_PATH=/root/autodl-fs/wyn/models/pi05_libero_finetuned_v044
export TOKENIZER_PATH=/root/autodl-fs/wyn/models/google/paligemma-3b-pt-224
export DATASET_ROOT=/root/autodl-fs/wyn/datasets/libero
export OUTPUT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only
export LOG_ROOT=/root/autodl-tmp/logs/vlm-only
export TMPDIR=/root/autodl-tmp/tmp

RUN_NAME=pi05-finetuned-vlm1-seed0-nocompile \
SAVE_ONLY_FINAL=true GPU_IDS=0,1 BATCH_SIZE=32 VLM_PROMPT_TOKENS=1 FLOW_STEPS=3000 \
bash examples/training/train_pi05_vlm_only.sh finetuned 0 \
  --policy.compile_model=false \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true

# 同一终端，完成后测四个 suite（新终端先重新设置上面的路径变量）。
RUN_NAME=pi05-finetuned-vlm1-seed0-nocompile \
CKPT_ROOT="$OUTPUT_ROOT" BASE_OUTPUT=/root/autodl-tmp/eval/vlm-only \
GPU_ID=0 VLM_PROMPT_TOKENS=1 CHECKPOINT_STEP=3000 \
bash run_eval_libero_vlm_only.sh finetuned trained all 0
```

`trained` 模式在没有 `000000` 时直接验证最终 checkpoint 的 prompt、训练来源及处理器，
v044 的归一化仍须与源模型一致。有 `000000` 时仍做跨 checkpoint 检查。
本次未保存真实随机初始化，不能选择 `initial` 或包含它的 `all`；当前 `baseline` 入口
也需要 step-0 处理器，因此上述最终模型使用 `trained` 模式评估。

## 三组对照与评测

每个来源均对比：

| 条件 | 权重与处理器 |
| --- | --- |
| baseline | 原始 source 权重、零 prompt、与该 run 的 step-0 相同的处理器和 dtype |
| initial | 该 run 保存的真实 step-0 随机 VLM prompt |
| trained | 该 run 的 step-3000 学习后 VLM prompt |

base 官方 processor 的 normalizer `features` 为空、未附统计量，直接原样加载的处理流程
与之前 ablation 训练时不同。因此 base 训练沿用 dataset QUANTILES；v044 保留其官方
MEAN_STD 统计量。**每个来源的三组对照内部保持数据处理一致**。
由原权重和 step-0 处理器组合生成的 baseline 是数据处理一致的零 prompt 对照，
不能直接沿用之前 base 原样评测的 2% 作为它的结果。两个来源之间仍有归一化方式差异，
绝对成功率的差异不能全部归于初始权重。

baseline 生成不加载/修改 source 权重，只在该 run 的 `matched_baseline/pretrained_model`
创建指向原权重的符号链接并复制小型配置/处理器文件；原模型目录需要保留。
baseline 和 initial 评测需要该 run 已保存 `000000`；trained 需要所选训练步的 checkpoint，
并在 `000000` 存在时额外比较初始化配置和统计量。
评测前检查 prompt 张量、源路径、配置、处理器及统计量是否匹配。

训练完成后，分别在两个终端评测，使用两张卡：

```bash
# 终端 1
GPU_ID=0 bash run_eval_libero_vlm_only.sh base all all 0

# 终端 2
GPU_ID=1 bash run_eval_libero_vlm_only.sh finetuned all all 0
```

也可以 `GPU_ID=0 bash run_eval_libero_vlm_only.sh both all all 0` 单卡顺序测完两来源的六组条件。
默认每个模型四套 LIBERO、每任务 10 次，共 400 episodes；环境 seed=1000，执行动作数=10。
输出根目录：`/root/autodl-tmp/eval/vlm-only`，每个来源/run/condition 独立保存并支持原评测续跑。
修改 token 数后，评测也传入同一个 `VLM_PROMPT_TOKENS`；不会在评测时修改学习好的 prompt 大小。

```bash
VLM_PROMPT_TOKENS=16 GPU_ID=0 bash run_eval_libero_vlm_only.sh base all all 0
CHECKPOINT_STEP=1000 GPU_ID=0 bash run_eval_libero_vlm_only.sh base trained all 0
```

主要报告 `trained − baseline`；`initial − baseline` 刻画插入随机 prompt 的影响，
`trained − initial` 刻画学习带来的改善。随机或全零 prompt 都不保证保持原函数。
不要将同环境 seed 理解成不同模型拥有完全相同的去噪噪声流。
先固定最终 3000 步作为主比较点，再把中间 checkpoint 作为曲线分析，避免挑最高测试结果。

官方 base 处理器：
https://huggingface.co/lerobot/pi05_libero_base/blob/main/policy_preprocessor.json
