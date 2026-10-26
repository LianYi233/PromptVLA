# 固定 P1/D0/W0/E1/T1，只替换 π0.5 预训练权重

这里不是把整个训练统一设为 seed1。固定 **2个 VLM prompt（4096参数）**：P1 读取同一个已保存的初始张量，D0/W0 固定旧有限样本计划，E1/T1 固定 occurrence 绑定的动作噪声和插值时间；剩余全局训练 seed=0，诊断探针 seed=2026。

| 名称中的来源 | 原始权重 | 目的 |
|---|---|---|
| libero | `/home/wyn/models/lerobot/pi05_libero_base` | 独立重复旧 W0-Bocc 70.50% 的设置 |
| generic | `/home/wyn/models/lerobot/pi05_base` | 在相同 LIBERO 设置下测试通用起始权重 |

历史70.50%是参考，不保证新训练精确复现。新 keyframe B1 已得到69.25%；应使用本次同时训练的 libero 组作为 generic 的直接对照。只改变权重来源也不能排除数值非确定性。

两组只训练VLM prompt，其余参数冻结；两卡/模型，每卡batch32，全局64，bf16，compile=false，3000步，只保存完整003000检查点。学习率、1000/30000原调度配置、50步连续动作目标、原固定探针、完整PDET输入追踪均沿用旧order-binding配方。**没有关键动作加权或课程重采样**；本比较用旧原始损失路径，不运行keyframe专用mask诊断，且不修改keyframe实验代码。

## 为什么不能直接替换 `--policy.path`

官方通用模型的config包含不同的相机键和32维状态/动作。直接加载整套配置，会同时改变数据接口及处理器，破坏单变量对照。准备工具创建两个小型适配目录：**config及所有processor文件均逐字节复制自同一个本地LIBERO来源，model.safetensors分别软链接到各自原始权重**。训练时两组都沿用原base配方的LIBERO数据集QUANTILES统计，而非通用模型的默认统计。不混入LIBERO模型的任何权重，不复制数GB权重，也不修改原始模型目录。

准备会校验模型架构及权重键/形状，记录两个原始权重文件的完整SHA256和原始config，要求P1张量及旧计划与参考运行一致。它只支持本项目可严格加载的LeRobot PI05 safetensors格式；不是OpenPI/JAX格式转换器。不能用简单改文件夹名冒充另一种模型。文件校验不同不等同于证明所有参数在bf16转换后都不同。

## 一次准备（CPU，无训练）

若通用模型尚未下载，在服务器执行（已下载则跳过）：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download lerobot/pi05_base \
  --include config.json model.safetensors \
  --local-dir /home/wyn/models/lerobot/pi05_base
```

在仓库根目录执行一次，不要在两个训练终端重复准备：

```bash
cd /home/wyn/VLAA/PromptVLA
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_source_compare prepare
```

默认输入为上述两个模型、`/home/wyn/datasets/libero`、`/data1/wyn/prompt-work/pi05-base-vlm2-init-seed1.safetensors`、`/data1/wyn/prompt-work/keyframe-r1/D0-W0.npz`。默认参考运行是新69.25%的 `pi05-key-r1-B-W0-Bocc-base-vlm2-P1-D0-E1-T1`，其W0计划与噪声bank此前已与旧70.50%设置核对。旧运行及完整PDET记录还在时可显式传入：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_source_compare prepare \
  --reference-run /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-orderbind-r1-W0-Bocc-base-vlm2-P1-D0-E1-T1
```

**上面两条prepare二选一**。模型路径不同时用 `--generic /实际/pi05_base目录`、`--libero ...`。已有输出拒绝覆盖；准备完成后复用 `check`。旧W0计划不可替换成FrameSkip U计划，P1不可重新随机生成。

输出 `/data1/wyn/prompt-work/source-pair-r1/` 中含 `libero/`、`generic/`、只读使用的 `D0-W0.npz`、`P1.safetensors` 和 `pair.json`。模型大文件只是软链接，训练及后续审计需要保留原始文件。首次完整权重哈希可能耗时数分钟；后续启动只核对大小、mtime、路径和小文件哈希，可选 `check --full-hash` 重新完整校验。

## 四卡并行训练

终端A（GPU0、1，重复LIBERO来源）：

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=0,1 MAIN_PROCESS_PORT=29581 REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_source_compare.sh libero
```

终端B（GPU2、3，通用来源）：

```bash
cd /home/wyn/VLAA/PromptVLA
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES \
GPU_IDS=2,3 MAIN_PROCESS_PORT=29582 REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_source_compare.sh generic
```

W&B默认打开，project=`prompt-learning`，online，不上传artifact。可追加 `--wandb.*=...`；其他训练参数不可随意覆盖。`DRY_RUN=true`可预检。两组名称为：

```text
pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1
pi05-source-r1-generic-W0-Bocc-base-vlm2-P1-D0-E1-T1
```

末尾`base`是沿用base训练/归一化配方的兼容标签，真正的权重来源由前面的`libero/generic`及`source_pair.json`区分。默认模型根目录 `/data1/wyn/chkpt/2601-lerobot/PromptVLA`，日志 `/data1/wyn/chkpt/2601-lerobot/logs`。目录已存在会拒绝覆盖；再次完整重复用新的`REPEAT_ID`，该ID仅用于命名，不改变随机因素。

## 审计和测试

两组完成后：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_source_compare audit \
  /data1/wyn/prompt-work/source-pair-r1 \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1 \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-source-r1-generic-W0-Bocc-base-vlm2-P1-D0-E1-T1
```

完整检查两组所有3000步的样本与ε/t哈希、初始化、训练配方、代码版本、最终processor统计，仅允许来源路径不同。还与参考运行核对实际样本/噪声输入；**不宣称与历史代码完全等价或GPU数值完全确定**。最终评估沿用每任务10次、seed1000、执行10步动作、四suite共400次，各在一个终端运行：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
GPU_ID=0 REPEAT_ID=1 EVAL_SEED=1000 EPISODES_PER_TASK=10 DRY_RUN=false \
bash run_eval_libero_source_compare.sh libero all
```

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
GPU_ID=2 REPEAT_ID=1 EVAL_SEED=1000 EPISODES_PER_TASK=10 DRY_RUN=false \
bash run_eval_libero_source_compare.sh generic all
```

测试保持一条原地更新的进度条和最终成功率。`both all`会在单卡顺序评估两组，不是两卡同时评估。
