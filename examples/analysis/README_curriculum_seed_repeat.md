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

`jobs.json` 保存六个名称、所有命令和分配的 GPU。已完成且通过最终模型/全步轨迹/运行时校验的模型会跳过；部分输出不会覆盖、删除或假装恢复。失败会停止本轮和后续轮次。每次只存最终模型，因此未完成训练没有可自动恢复的中间 checkpoint。运行期间不要更新代码或依赖，以免不同组使用不同实现。

完成后自动写 `audit.json`，亦可手动重跑：

```bash
bash examples/training/train_pi05_curriculum_seed_repeat.sh audit
```

检查每组内部真实使用的每一个样本/ε/t 配对、相同训练配置/代码、不同课程顺序，以及两组之间确实不同的 prompt 与随机输入。该检查不宣称逐位数值确定性，也不宣称与历史代码完全等价。后续评价应使用与原 D/E/F 相同的评估 seed=1000、每 task 10 episodes 和 PRO 扰动维度；对每套种子分别计算 D−F、D−E，再汇总三套训练结果，不把每个 rollout 当作独立的训练重复。
