# D/E/F 两套新种子重复实验

保持原 D/E/F 的训练方法与预算：`pi05_libero_base`、VLM prompt 2 tokens + **完整 Action Expert** 联合训练、easy/hard **50/50**、6000 updates、每模型 2 GPU × batch32（global64）、bf16、关闭 compile、原学习率与调度、只保存最终 `006000`。每个模型独立加载预训练权重，不接着上一个模型训练。

| 轮次 | GPU 0,1 | GPU 2,3 | 顺序 |
|---|---|---|---|
| 1 | D1 | D2 | 难→易 |
| 2 | E1 | E2 | 易→难 |
| 3 | F1 | F2 | 随机 |

`1/2` 是重复组标签，实际随机种子默认是 **2/3**，避免复用原实验中的 0/1：

| 设置 | 原 D/E/F | D1/E1/F1 | D2/E2/F2 |
|---|---:|---:|---:|
| prompt 初始化 P | 1 | 2 | 3 |
| 基础样本池和任务槽位 D | 0 | 2 | 3 |
| flow noise ε 的 E | 1 | 2 | 3 |
| flow time t 的 T | 1 | 2 | 3 |
| 全局训练 seed | 0 | 2 | 3 |
| 重采样/随机课程 S | 0 | 2 | 3 |
| 有效秩固定探针 seed | 2026 | 2026 | 2026 |

W0 和 Bocc 协议保持不变。同一组三个模型共享同一 prompt 初值、相同 occurrence 池、任务槽位与每次样本使用绑定的 ε/t，只改变每个 task 内的训练顺序。两组之间改变随机性，样本池与任务槽位允许不同。评分文件及评分方法不变，不重新跑 GPU 离线评分。候选池仍按原方法先生成 3000×64 次使用，再重采样为 6000×64 次使用；没有把候选预算偷偷增加一倍。

新 P2/P3 通过原生模型构造的初始化导出模式自动生成，0 次更新；仅导出几 KB 的初始 prompt，不保存完整初始模型。已有文件只有通过 seed/source/hash 验证才复用。没有把旧 P1 重命名为 P2/P3。

## 启动

在训练用 `lerobot` 环境、四张 GPU 空闲时：

```bash
cd /home/wyn/VLAA/PromptVLA
git pull --ff-only origin feat/keyframe-loss-curriculum

mkdir -p /data1/wyn/chkpt/2601-lerobot/logs
nohup env DRY_RUN=false GPU_IDS_A=0,1 GPU_IDS_B=2,3 \
  PORT_A=29592 PORT_B=29593 SEED_SET_1=2 SEED_SET_2=3 \
  bash examples/training/train_pi05_curriculum_seed_repeat.sh run \
    --wandb.enable=true --wandb.project=prompt-learning \
    --wandb.mode=online --wandb.disable_artifact=true \
  > /data1/wyn/chkpt/2601-lerobot/logs/curriculum-def-seeds-2-3-launch.log 2>&1 &
```

一次启动自动完成三轮六组。初始化导出在训练前进行；训练时两组并行、轮次之间有同步屏障。旧的 `RUN_NAME`、`PROMPT_INIT_PATH`、`TRAIN_SCOPE` 等变量不会覆盖本实验。步数、训练范围、batch、保存策略固定，不支持用额外 CLI 修改；仅接受 `--wandb.*=...`。

可设置 `OUTPUT_ROOT`、`LOG_ROOT`、`SOURCE_PAIR_DIR`、`SCORES_PATH`、`REFERENCE_RUN`、`DATASET_ROOT`、`TOKENIZER_PATH`、`SEED_WORK_DIR`、`SEED_LOG_DIR`；默认沿用本机路径。`REFERENCE_RUN` 是原 source-pair 的 LIBERO 参考训练，用于核查权重和评分来源，不是继续训练的起点。

启动前可 `DRY_RUN=true bash examples/training/train_pi05_curriculum_seed_repeat.sh run`：检查 CPU 输入并生成匹配计划，不构造模型、不导出初值、不启动训练。若只想列出命令和名称，用 `...sh list`，不读取数据/模型。

## 命名、日志、恢复

例如 D1：

```text
pi05-curriculum-D1-H2E-e50-h50-S2-steps6000-aexpert-libero-Bocc-vlm2-P2-D2-E2-T2
```

D2 相应为 `D2`、`S3`、`P3-D3-E3-T3`；E 为 E2H，F 为 RND。最终模型都在：

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/<模型名>/checkpoints/006000/pretrained_model
```

计划和微小初始化文件：`/data1/wyn/prompt-work/curriculum-def-seeds-2-3/`。

```bash
tail -f /data1/wyn/chkpt/2601-lerobot/logs/curriculum-def-seeds-2-3-launch.log
cat /data1/wyn/chkpt/2601-lerobot/logs/curriculum-def-seeds-2-3/status.json
# 每个 cell 的训练输出：
tail -f /data1/wyn/chkpt/2601-lerobot/logs/curriculum-def-seeds-2-3/D1.launch.log
```

`jobs.json` 保存六个名称、所有命令和分配的 GPU。已完成且通过最终模型/全步轨迹/运行时校验的模型会跳过；默认拒绝覆盖部分输出。失败会停止本轮和后续轮次，并在主日志显示失败子进程的最后输出。每次只存最终模型，因此未完成训练没有可自动恢复的中间 checkpoint。运行期间不要更新代码或依赖，以免不同组使用不同实现。

### 上次失败、目录已存在时

先确认上一个调度程序已退出。重新启动时在 `nohup env` 后加入 **`RESTART_INCOMPLETE=true`**：

```bash
LAUNCH_LOG="/data1/wyn/chkpt/2601-lerobot/logs/curriculum-def-recovery-$(date +%Y%m%d-%H%M%S).log"
nohup env DRY_RUN=false RESTART_INCOMPLETE=true \
  GPU_IDS_A=0,1 GPU_IDS_B=2,3 PORT_A=29592 PORT_B=29593 \
  SEED_SET_1=2 SEED_SET_2=3 \
  bash examples/training/train_pi05_curriculum_seed_repeat.sh run \
    --wandb.enable=true --wandb.project=prompt-learning \
    --wandb.mode=online --wandb.disable_artifact=true \
  > "$LAUNCH_LOG" 2>&1 &
tail -f "$LAUNCH_LOG"
```

脚本先检查全部六组，再把**没有最终权重**的未完成输出移动到
`OUTPUT_ROOT/_incomplete_backups/<原名称>--<时间戳>/run/`，保留旧日志、状态和恢复记录；不删除任何模型。
这些组从同一预训练模型、同一初始 prompt、同一计划重新训练6000步，而不是从中间步继续。
P2/P3 `.safetensors` 及计划文件保持原位，验证通过后直接复用。
仍有训练进程使用该目录、输出目录是软链接、或已有最终权重但校验未通过时，都不会自动备份重训。
`DRY_RUN=true RESTART_INCOMPLETE=true` 仅显示将备份哪些目录，不移动它们。

完成模型可跨本次**已验证的恢复/审计代码改动**复用，但模型、采样、初始化、训练参数生成或依赖版本的变化仍然拒绝混用。这一恢复选项解决目录阻塞，不等于已经确定此前训练中断的根因；旧的错误末尾会在主日志显示，完整记录保存在备份内。

完成后自动写 `audit.json`，亦可手动重跑：

```bash
bash examples/training/train_pi05_curriculum_seed_repeat.sh audit
```

检查每组内部真实使用的每一个样本/ε/t 配对、相同训练配置/代码、不同课程顺序，以及两组之间确实不同的 prompt 与随机输入。该检查不宣称逐位数值确定性，也不宣称与历史代码完全等价。后续评价应使用与原 D/E/F 相同的评估 seed=1000、每 task 10 episodes 和 PRO 扰动维度；对每套种子分别计算 D−F、D−E，再汇总三套训练结果，不把每个 rollout 当作独立的训练重复。

## 训练完成：六组 LIBERO-PRO 四卡评测

使用此前能够运行 PRO 的 `promptvla-eval` 环境。专用入口识别 D1/E1/F1（训练种子2）和 D2/E2/F2（训练种子3）；所有模型的**评估 seed 都是1000**。复用原 A–H 评测器、checkpoint/processor 校验、task-level 断点续测和完整结果校验，不修改底层仿真协议。

```bash
conda activate promptvla-eval
cd /home/wyn/VLAA/PromptVLA

export CKPT_ROOT=/data1/wyn/chkpt/2601-lerobot/PromptVLA
export BASE_OUTPUT="$CKPT_ROOT/eval/curriculum-def-seeds-2-3"
export EVAL_LOG_ROOT=/data1/wyn/chkpt/2601-lerobot/logs/eval/curriculum-def-seeds-2-3
export BENCHMARK_ROOT=/home/wyn/VLAA/LIBERO-PRO
export TOKENIZER_PATH=/home/wyn/models/paligemma-3b-pt-224
export SEED_SET_1=2 SEED_SET_2=3
export GPU_IDS=0,1,2,3 CELLS="D1 D2 E1 E2 F1 F2"
export CHECKPOINT_STEP=6000 EVAL_SEED=1000 EPISODES_PER_TASK=10
export PRO_DIMENSIONS=object,swap,lan,task PROGRESS=true DRY_RUN=false
mkdir -p "$EVAL_LOG_ROOT"

nohup bash run_eval_libero_curriculum_seed_repeat_pro.sh run \
  >> "$EVAL_LOG_ROOT/pro-launch.log" 2>&1 < /dev/null &
echo "PRO queue PID=$!"
```

先逐一核查六组最终checkpoint、保存的训练P/D/E/T种子、6000步、batch32和Bocc绑定，再检查PRO资源，全部通过后才启动GPU。GPU0依次D1→F1，GPU1依次D2→F2，GPU2运行E1，GPU3运行E2，每卡同一时间一个模型。四类扰动×四个suite×10任务×10episodes = **1600 episodes/模型，六组9600 episodes**；本次不含env扰动，不是完整五类PRO。

同一终端已导出上述变量时，用原地刷新监控（Ctrl+C只退出监控）：

```bash
watch -n 10 -t 'bash run_eval_libero_curriculum_seed_repeat_pro.sh status'
```

模型载入阶段可能暂时没有episode进度；预检阶段显示pending/preflight。新终端查看时应重新设置上述export变量，或使用默认路径和种子并清除遗留的`CELLS/BASE_OUTPUT/EVAL_LOG_ROOT`等覆盖值。

```bash
bash run_eval_libero_curriculum_seed_repeat_pro.sh summary
```

输出同原D/E/F格式的suite表及扰动类别表，保存在：

```text
/data1/wyn/chkpt/2601-lerobot/logs/eval/curriculum-def-seeds-2-3/
  pro-batch-seed1000-ep10-dimsobject-swap-lan-task/
    D1.preflight.log / D1.progress.log / D1.result.txt  （其他五组同理）
    status.json / summary.txt / summary.json
```

完整逐任务记录在`BASE_OUTPUT/pro/seed1000-ep10-dimsobject-swap-lan-task/`。缺失/部分结果会显示未完成而不是0%；summary在有缺失时返回非零退出码。失败后等旧队列退出，重跑同一启动命令可复用已完成任务；例如`CELLS="F1 F2" GPU_IDS=0,1`只重试两组，汇总六组前恢复完整CELLS。`DRY_RUN=true`仅预检。不要把原A–H入口的CELLS改为D1等标签，它只识别A–H。
