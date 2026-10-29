# 97.75% 模型的四轮消融

参考模型为 `pi05-curriculum-H2E-e30-h70-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1`。
一个入口自动执行四轮，每轮两组并行，两组完成并通过审计后才进入下一轮。
每组都从 `source-pair-r1/libero`（原始 `pi05_libero_base` 权重）重新初始化，
导入同一个 P1；不会接着上一组训练好的模型继续微调。

| 轮次 | GPU0、1 | GPU2、3 |
| --- | --- | --- |
| 1 | B：E2H，简单30%／困难70%，prompt+expert | C：task内随机，30%／70%，prompt+expert |
| 2 | D：H2E，50%／50%，prompt+expert | F：task内随机，50%／50%，prompt+expert |
| 3 | E：E2H，50%／50%，prompt+expert | A：H2E，30%／70%，prompt+expert，重跑参考设置 |
| 4 | G：H2E，30%／70%，冻结P1，仅训练expert | H：H2E，30%／70%，仅训练prompt，冻结expert |

默认新实验编号为 **r2**，避免覆盖已有r1模型。`REPEAT_ID`仅用于命名，不改变随机种子。
固定VLM2、P1/D0/E1/T1、Bocc、S0、全局seed0、双卡×batch32、bf16、关闭compile。
保持原来的学习率、warmup、固定探针及诊断设置，默认每组6000步。
G保留两个P1 token参与前向传播；它没有删除token，也不将它们清零。

## 一次启动四轮

更新 `feat/keyframe-loss-curriculum` 后，在原训练环境运行：

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
mkdir -p /data1/wyn/chkpt/2601-lerobot/logs

nohup env REPEAT_ID=2 FLOW_STEPS=6000 \
  GPU_IDS_A=0,1 GPU_IDS_B=2,3 PORT_A=29588 PORT_B=29589 \
  bash examples/training/train_pi05_curriculum_ablation.sh \
    --wandb.enable=true \
    --wandb.project=prompt-learning \
    --wandb.mode=online \
    --wandb.disable_artifact=true \
  > /data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-launch.log 2>&1 &
```

不需要手动启动第二个终端，不需要在轮次之间再次运行命令。`nohup`使任务在终端断开后继续运行。
训练过程中不要切换仓库版本；跨组审计要求这些新实验使用相同的训练代码。

可选CPU检查：

```bash
DRY_RUN=true REPEAT_ID=2 FLOW_STEPS=6000 \
  bash examples/training/train_pi05_curriculum_ablation.sh
```

正式启动已经内置同样的全量检查，无需先执行DRY_RUN。检查阶段会复用或准备全部采样计划，
不会加载模型上GPU，不会创建训练输出目录。保留原 `source-pair-r1`、评分缓存
`frameskip-r1/scores.npz` 和原source参考run即可，不需要重新评分。
原参考run默认为 `pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1`；
其配置、处理器和PDET记录用于核对训练配方。

## 保存、日志与中断

每个实验仅保存 `checkpoints/006000/` 的完整模型和训练状态；不保存000000或中间模型。
`checkpoints/last`只是指向006000的软链接，不复制第二份权重。
轻量的prompt/rank探针、PDET输入记录、训练日志仍保留，用于分析和审计。
八组各保留一个最终模型，不会删除前一组或历史r1模型。

```bash
# 总进度：预检、每轮开始、每组完成/跳过
tail -f /data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-launch.log

# 单组启动、训练及审计输出；B换成C/D/F/E/A/G/H
tail -f /data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-steps6000/B.launch.log
```

日志根目录还保留每组 `<完整RUN_NAME>.log`。总目录中：

- `jobs.json`：八组完整名称、GPU、计划路径与命令。
- `status.json`：pending/running/verifying/complete/failed等状态。
- `runner.log`：总进度记录。
- `audit.json`：全部训练完成后的跨组审计。
- `B.preflight.log` 等：每组CPU预检结果。

训练错误会终止本轮仍在运行的另一组及后续轮次，避免静默漏跑实验。
重新执行相同入口时，仅跳过最终checkpoint完整且审计通过的实验。
已有但未完成的目录会在训练前报错；脚本不会覆盖、删除或假装自动续训它。
只保存最终模型意味着6000步之前中断时没有中途checkpoint可供续训。

### 6000步结束后因路径别名审计失败的恢复

如果B组已输出 `End of training`，随后审计报告 `recipe` / `dataset_root` 不同，
先核对具体字段。已修复 `/home/wyn/...` 软链接与 `/data/...` 真实路径被误判的问题：
仅在两个路径都是已存在的本地目录，且 `samefile()` 确认指向同一目录时，
规范化 dataset_root、recipe.dataset.root、recipe.policy.tokenizer_name；其他差异仍拒绝。
审计报告保留原路径和解析结果，失败时打印具体字段和值。不会修改旧manifest。

B有完整006000、C被停止且没有最终模型的情况，更新代码后运行：

```bash
conda activate lerobot
cd /home/wyn/VLAA/PromptVLA
git pull --ff-only origin feat/keyframe-loss-curriculum

python examples/analysis/recover_curriculum_ablation.py --archive-cell C
```

恢复脚本先锁定队列、检查旧进程组已退出，并重新核验B的最终checkpoint、全步输入、
配方及代码兼容性。验证通过后保留B，把C移到输出根目录的 `.interrupted/` 时间戳目录，
保留其PDET/rank记录及旧训练日志；不删除模型，不启动训练。若C已有任何最终模型文件，
脚本会停止要求检查，不会把可能已保存的模型归档。可用 `--directory` 指定其他队列日志目录。

这次修复只调整审计逻辑。复用旧B时，代码检查会重建两个Git版本的完整源文件哈希，
要求与各自PDET记录一致，再确认采样/训练/启动代码未变；只允许指定审计函数和独立审计工具变化。
模型、优化器、采样计划生成或启动参数变化仍拒绝；无法验证的历史未提交修改也拒绝。
`curriculum_runtime_audit.json` 与最终suite审计保留兼容性记录，不将它等同于数值确定性。

随后使用**相同r2编号**重新启动；追加总日志，不覆盖上次报错：

```bash
nohup env REPEAT_ID=2 FLOW_STEPS=6000 \
  GPU_IDS_A=0,1 GPU_IDS_B=2,3 PORT_A=29588 PORT_B=29589 \
  bash examples/training/train_pi05_curriculum_ablation.sh \
    --wandb.enable=true --wandb.project=prompt-learning \
    --wandb.mode=online --wandb.disable_artifact=true \
  >> /data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-launch.log 2>&1 &
```

队列再次核验并跳过B，只重跑C；C完成后自动接第2、3、4轮。每组仍只保留最终006000。

## 名称和匹配规则

默认输出根目录：`/data1/wyn/chkpt/2601-lerobot/PromptVLA`。
命名格式：

```text
pi05-curriculum-<顺序>-e<简单百分比>-h<困难百分比>-S0-steps6000-<训练范围>-r2-libero-Bocc-base-vlm2-P1-D0-E1-T1
```

| 标记 | 含义 |
| --- | --- |
| E2H / H2E / RND | task内升序 / 降序 / 随机顺序 |
| aexpert | VLM prompt + 完整action expert都训练 |
| frozenp-aexpert | 保留并冻结P1，只训练完整action expert |
| promptonly | 仅训练VLM prompt |

同一比例下，三种顺序共享**完全相同的使用实例池**，只改变每个task内部的顺序。
随机顺序使用独立的局部NumPy生成器，不消耗训练的全局随机状态。
每次使用对应的ε/t随occurrence ID移动。所有实验固定task出现位置。
G/H与新A共享同一个H2E30计划；G最终prompt张量必须与初始P1逐元素完全相同。
跨比例组的使用池有意不同，不能将跨比例比较描述为纯顺序干预。

保持原E2H/H2E计划生成逻辑、数组校验值和旧r1名称不变。新run相对历史模型的代码版本不同，
历史参考审计会如实报告不同版本，不据此声称数值确定性或历史代码等价。

## 评估与独立审计

评估入口同步支持随机顺序和三种训练范围。例如测试B：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES \
CURRICULUM_DIRECTION=easy_to_hard EASY_PERCENT=30 TRAIN_SCOPE=joint \
SAMPLE_SEED=0 REPEAT_ID=2 FLOW_STEPS=6000 \
GPU_ID=0 EVAL_SEED=1000 EPISODES_PER_TASK=10 PROGRESS=true \
bash run_eval_libero_ratio_curriculum.sh all
```

C/F用 `CURRICULUM_DIRECTION=random`；G用 `TRAIN_SCOPE=expert_only`；H用 `TRAIN_SCOPE=prompt_only`。
其他比例和方向按表设置，评估会校验名称、训练配置及冻结prompt的最终值。

全部训练完成后总入口自动跨组审计；也可单独执行：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" REPEAT_ID=2 FLOW_STEPS=6000 \
python -m lerobot.scripts.pi05_curriculum_ablation audit
```

每一步的实际sample/occurrence/噪声hash均参与核对，不仅比较命令中的seed。
审计通过代表受控输入和配方符合设计，不代表GPU数值绝对确定。
