# 可调简单/困难占比的双向课程

本实验默认训练 **2个VLM prompt token + 整个action expert**，起点仍为 `pi05_libero_base`。
视觉编码器和VLM主干冻结。动作Transformer、动作输入/输出投影、时间MLP参与训练；action prompt=0。
P1、全局seed0、E1/T1、Bocc、双卡×每卡batch32、原学习率/损失均保留。
现在默认 `FLOW_STEPS=6000`，只保存最终完整006000；可显式调整训练步数。
继承同样的rank/flow探针；rank仍指对prompt的Jacobian，不是整个动作网络的秩。

新增四轮消融入口见 [自动运行8组课程/模块对照](README_pi05_curriculum_ablation.md)。
单组入口现在也支持 `CURRICULUM_DIRECTION=random`（默认easy50%）、
`TRAIN_SCOPE=joint|expert_only|prompt_only`（默认joint）；expert_only保留并冻结原P1。
E2H/H2E的原默认参数和计划不变，下面的双向示例仍为joint。

## 旧版与新版的区别

旧 `train_pi05_hard_to_easy.sh` 按每个task内的评分降序重排原192,000次样本使用，
每次原有使用保留一次。它没有显式二分简单/困难，也没有对难样本增加采样次数。
旧脚本与已准备的 `hard2easy-r1/H2E.npz` 保持原行为。

新脚本 `train_pi05_ratio_curriculum.sh` 同时控制**顺序和总曝光比例**：

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `CURRICULUM_DIRECTION` | 每个task内由易到难/由难到易 | `easy_to_hard` |
| `FLOW_STEPS` | 总训练步数；计划、审计、最终保存及评估同步使用 | 6000 |
| `EASY_PERCENT` | 全部样本使用中简单组的占比，整数1–99 | 易→难70；难→易30 |
| `SAMPLE_SEED` | 组内抽样/重复采样的独立种子，0–2³²−1 | 0 |
| `REPEAT_ID` | 重复实验编号，只改变输出名称 | 1 |
| `ORDER_DIR` | CPU采样计划目录 | 按顺序、比例、S、步数自动区分 |

`EASY_PERCENT`始终指**简单样本**，不随方向改变含义。例如难→易且`EASY_PERCENT=20`是80%难、20%易。
这是启动时可调的实验参数，不会热修改正在训练的计划，也不是每个batch始终混合70:30，
更不是loss乘以0.7/0.3。为了可审计，已有目录与新参数不一致时直接拒绝；不会静默覆盖。

## 定义和训练过程

1. 复用 `source-pair-r1/D0-W0.npz` 的原候选样本使用池及旧评分缓存。
   根据 `FLOW_STEPS×64` 循环延长原task-slot序列，必要时截断；6000步恰好重复两次原task序列。
   不扩大候选帧集合，不重新GPU评分，不加载generic模型训练。
2. 在每个task内部，将候选池中**不同的帧/chunk起点**按 `score` 升序排列，低分一半为简单组，
   高分一半为困难组；数量为奇数时简单组取向下整。相同分数按全局frame index打破平局。
   同一帧不会因重复使用而同时被划入两组。完全同分的task报错，因为无法提供有意义的难度区分。
   边界同分情况记录在manifest的`boundary_score_tied`；这些标签只是可复现的代理划分。
3. 在每个task的新使用次数预算内分配简单/困难配额。采用最大余数法，使全局计数达到目标，
   每个task的舍入误差小于一次使用。相同步数的两个方向共享相同task曝光总量和slot序列。
4. 各组使用固定`SAMPLE_SEED`洗牌循环：一轮内无放回，候选用完后再开始新一轮；最后一轮截断。
   因而重点组有更多重复训练，另一组有较少使用；严格遵守指定总步数。抽样对象是原候选池的使用实例，
   若原池本身重复某帧，该帧仍保留原候选频次。
5. 为选出的每次使用生成独立occurrence ID。先生成与方向无关的规范使用池，再按对应方向放回
   固定task slots。每个task内部评分单调上升/下降；平分边界先按简单/困难组、再按ID打破平局。
   连续50步动作目标不被拆开。多个task混在batch里，全局batch平均分不保证逐步严格单调。

默认6000步，两组各有384,000次使用，总量为：

| 实验 | 顺序 | 简单组使用次数 | 困难组使用次数 |
| --- | --- | --- | --- |
| E2H-e70-h30 | 先易后难，易组训练更多 | 268,800 | 115,200 |
| H2E-e30-h70 | 先难后易，难组训练更多 | 115,200 | 268,800 |

直观上约前70%的task内使用集中在先训练的重点组，最后30%进入另一组；各task在全局位置上交错，
不是要求所有task同时在第4200步切换。对整个6000步只排序一次，**不会在3000步后重启课程**。
`schedule.csv`记录每步实际简单/困难次数、平均/最低/最高分。
`curriculum.json`记录逐task配额、候选/实际唯一帧数、重复使用数量、评分recipe和输入哈希。

评分仍是旧FrameSkip风格的**难度/关键性代理**：默认0.6动作变化+0.2视觉变化+0.2夹爪事件邻域，
以现有缓存中保存的recipe为准。它不是模型实际预测误差，也不保证高分帧一定更难。

## 噪声如何处理、如何解释比较

仍用E1/T1和原CPU indexed bank规则；6000步扩展到384,000个key，不循环复用旧噪声。
**重复同一帧的不同使用分配不同ID和ε/t**，
不会让重复帧一直复用同一个噪声。每个新使用的噪声/时间跟随它移动（Bocc）。

由于采样配额变了，新使用池与历史W0/H2E不同，样本与噪声的配对也会变化。与历史prompt-only随机顺序
参考相比，干预包括重采样、顺序、解冻action expert和训练步数；不能把差异全部归因于顺序。
这两组之间共同保持动作网络训练范围，但同时改变顺序与比例。如果之后要单独判断顺序效应，
使用**相同FLOW_STEPS、EASY_PERCENT和SAMPLE_SEED**只换方向：实现会保证两个方向的使用池、每次使用对应的帧及ε/t完全一致。
相同参数不同REPEAT_ID只重复命名，不生成新的随机种子。

## 两个终端同时训练

先更新分支并激活训练环境，在 `/home/wyn/VLAA/PromptVLA` 下运行。
两组各用两卡，保持global batch64。设置`DRY_RUN=true`可先准备/检查CPU计划并查看完整命令。

终端一，GPU0/1，易→难，简单70%：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES -u ORDER_DIR \
CURRICULUM_DIRECTION=easy_to_hard EASY_PERCENT=70 SAMPLE_SEED=0 \
FLOW_STEPS=6000 \
TRAIN_ACTION_EXPERT=true GPU_IDS=0,1 MAIN_PROCESS_PORT=29584 \
REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_ratio_curriculum.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

终端二，GPU2/3，难→易，困难70%：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u PROMPT_INIT_PATH -u PROMPT_EXPORT_PATH \
    -u PROMPT_EXPORT_REFERENCE -u PDET_CASES -u ORDER_DIR \
CURRICULUM_DIRECTION=hard_to_easy EASY_PERCENT=30 SAMPLE_SEED=0 \
FLOW_STEPS=6000 \
TRAIN_ACTION_EXPERT=true GPU_IDS=2,3 MAIN_PROCESS_PORT=29585 \
REPEAT_ID=1 DRY_RUN=false \
bash examples/training/train_pi05_ratio_curriculum.sh \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

输出分别为：

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-curriculum-E2H-e70-h30-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1
/data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-curriculum-H2E-e30-h70-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1
```

默认数据/模型/tokenizer/日志路径与旧版一致。复用
`/data1/wyn/prompt-work/source-pair-r1`、`/data1/wyn/prompt-work/frameskip-r1/scores.npz`及已完成的
`pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1`参考；这些文件必须保留。
CPU计划分别在 `/data1/wyn/prompt-work/ratio-curriculum/E2H-e70-h30-S0-steps6000` 和 `H2E-e30-h70-S0-steps6000`。
不覆盖旧模型；已有相同名称输出时换REPEAT_ID。只保存完整006000，保留小型审计/探针文件。
`FLOW_STEPS=3000`保留之前不含steps后缀的名称和原计划，可继续检查/评估旧003000模型。
6000步是从原始预训练权重重新开始，不是接着旧003000恢复训练。
学习率仍为2.5e-5，warmup仍为1000步，decay配置仍为30000步；只增加更新次数，不缩放学习率日程。

## 评估与审计

训练结束自动审计全部配置步数的实际采样序列、配额、task slots、bank和训练配方。
与旧3000步参考比较共有的前192,000个bank key；新增key检查完整trace覆盖并保存哈希，明确标注无历史参考。
它允许已声明的重采样、顺序、训练范围及步数变化，并单独报告历史代码差异；不证明GPU数值确定性。
不用重新训练就可以重跑：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m lerobot.scripts.pi05_ratio_curriculum audit \
  /data1/wyn/prompt-work/ratio-curriculum/E2H-e70-h30-S0-steps6000 \
  /data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-curriculum-E2H-e70-h30-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1
```

训练结束后在两个终端用单卡分别评估。首组：

```bash
env -u PRETRAINED_PATH -u BASE_PRETRAINED_PATH -u RUN_NAME -u RUN_PREFIX \
    -u VLM_PROMPT_TOKENS -u BASE_CKPT -u BASE_MODEL_PATH -u PDET_CASES -u ORDER_DIR \
CURRICULUM_DIRECTION=easy_to_hard EASY_PERCENT=70 SAMPLE_SEED=0 \
FLOW_STEPS=6000 \
GPU_ID=0 REPEAT_ID=1 EVAL_SEED=1000 EPISODES_PER_TASK=10 \
PROGRESS=true DRY_RUN=false \
bash run_eval_libero_ratio_curriculum.sh all
```

第二组用同一命令，将 `CURRICULUM_DIRECTION=hard_to_easy EASY_PERCENT=30 GPU_ID=2`。
共4suite×10task×10episode=400次；原地进度条/ETA和最终成功率，完整输出进入日志。
如果训练时改过步数、比例或S/REPEAT_ID，测试必须传入相同值。评估自动选择对应最终checkpoint，检查并加载训练后的完整动作网络权重。

验证覆盖：CPU配额/排序/重复使用与独立ε/t、参数校验、完整训练和评估dry-run、全步trace审计。
未在开发环境运行A100训练或测量显存峰值。
