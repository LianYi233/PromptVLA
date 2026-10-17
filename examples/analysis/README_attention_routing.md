# 探索 vlm_only 的注意力去向

目标：先解释当前记录中图像以外约 98.76% 的注意力分配，再检验这些路径是否能解释
vlm_prompt_only 的 LIBERO 表现。这个比例属于当前样本及其层/head/query/去噪平均口径，
不是整个 LIBERO 的统计结论，也不是“98.76% 的信息来自非视觉”的证据。

## 分支与模型

`prompt-learning` 以远端 `piper` 的 `3f3fb664c608578b3d9693669368d9ed7644a302` 为基底，
包含 `examples/piper/eval_four_tasks.sh`、`examples/training/train_pi05_prompt_ablation.sh`
及其训练/离线评估依赖；再迁入 `prompt-ablation` 的双相机诊断代码
（截至 `0f81da7349143dee566d2c17ffd93f07a854f132`）。
检查时未找到标题为 `Update Piper evaluation and prompt ablation training` 的远端提交；
本分支保留的是上述远端 piper 提交中的脚本版本。

新分支不代表需要重新训练。默认继续评估同一 `pi05-vlm_only-seed0` 的 003000 checkpoint，
checkpoint 默认目录仍是原来的 `prompt-ablation` 训练目录。
以实际 checkpoint config 为准，核对 `num_vlm_prompt_tokens`、`num_prompt_tokens`、
`chunk_size`、`n_action_steps`、去噪次数、backbone、归一化统计、tokenizer 和权重来源。
脚本最后的 `0` 是训练 run seed；评估的环境/全局 seed 当前由评估配置默认设为 1000。
不要把切换训练 seed 当成在同一权重上更换环境 seed。

## 1. 首先做完整的注意力分组统计（已实现）

当前真实布局是：图像 patch → 文本化任务与状态 → VLM prompt → action prompt → action chunk。
动作 query 能读取全部有效 key；VLM prompt query 只读取 prefix。
这里的箭头表示 token 排列，不是信息传播方向。

| 分组 | 内容与检查 |
| --- | --- |
| `camera_0`, `camera_1`, … | 所有模型相机，包括未显示或 padded 的相机；padded 相机应为零 |
| `language_state` | 有效任务、离散状态、模板与特殊 token |
| `language_padding` | 文本 padding，必须严格为零 |
| `vlm_prompt` | 全部学习得到的 VLM prompt key |
| `action_prompt` | 专家 prompt；正确的 vlm_only 配置应为零个 token |
| `action_executed` | chunk 中前 `n_action_steps` 个动作 key |
| `action_future` | 同一 chunk 中其余动作 key，并不表示这些动作之后一定会执行 |

`Pi05PrepareStateTokenizerProcessorStep` 构造的是
`Task: ..., State: ...;\nAction: `，然后统一分词。
目前不凭字符串猜测任务/状态的 token 边界；后续用同一 tokenizer 的可靠 offset/special-token
映射拆为 task/state/template/special，跨边界 token 单列。原始 token ID 已保存，可用于核查。

设选定 query 数为 Q，某 head 的全 key softmax 为 A，则组 g 的 mass：

```text
M[g] = mean_over_queries(sum_over_keys_in_g(A[q,k]))
sum_g M[g] = 1
```

每次 prediction、每次去噪、每个 head 单独保存，容许浮点闭合误差 2e-6。
另报有效 key 数、每 key 平均 mass、相对“均匀分配给所有可见 key”的倍数。
后两项用于控制组大小；它们也不是因果重要性指标。

保存的 NPZ 字段（P=prediction，D=去噪 pass，H=query head，G=分组，K=全部 key）：

| 字段 | 形状 / 含义 |
| --- | --- |
| `routing_head_mass` | `[P,D,H,G]`；保留全部 head/pass，query 已平均 |
| `routing_visible_key_counts` | `[P,D,G]`；从实际 mask 计算每个 query 的平均可见 key 数 |
| `routing_token_mass` | `[P,D,K]`；平均 query 与 head 后的逐 key 概率 |
| `routing_key_group_ids` | `[P,K]`；每个 key 所属组，覆盖所有位置 |
| `routing_token_ids` | `[P,K]`；语言位置为实际 tokenizer ID，其余为 -1 |
| `routing_group_names` | 分组名称，空组保留 |

现阶段没有保存逐 query 的注意力或 value/output 投影，所以不能据此分析动作维度贡献。
一个 action token 对应 chunk 的一个时间位置，而非单个关节维度。
`ATTENTION_DENOISE=first/last` 只改变热图及摘要的展示选择；routing 原始数组始终保留全部 pass。
VLM prefix 每次 prediction 只算一次，因此其 D=1。

## 2. 首轮运行与交付

先在相同 checkpoint 上跑 libero_10 每任务 1 episode，验证统计，再扩展到所有 suite。
这是机制探索用的小样本，不用它重新估计公布的成功率。保存全部成功与失败 episode。

```bash
# 在已有 lerobot 环境、仓库根目录运行。沿用原 checkpoint 与字体路径即可。
GPU_ID=0 \
BASE_CKPT=/actual/run/checkpoints \
ATTENTION_FONT_PATH=/actual/times.ttf \
BASE_OUTPUT=/root/autodl-tmp/eval/prompt-learning-routing-l17-pilot \
EPISODES_PER_TASK=1 \
bash run_eval_libero_attention_routing.sh libero_10 0
```

每个 episode 会保存视频、双相机原图/heatmap，以及以下三个文件：

- `*.attention.routing_summary.json`：全组平均占比、逐 prediction 占比、闭合误差。
- `*.attention.routing.csv`：逐 prediction × pass × head × group 的 mass 和 key 数控制。
- `*.attention.routing_top_tokens.csv`：每次 prediction 的 top-20 key，带分组、组内位置及 token ID。

摘要同时报告所有 pass 平均和与当前热图一致的 pass 选择。top-20 文件采用后者，
更细的 pass 分析使用原始 NPZ。重新导出无需加载模型：

```bash
PYTHONPATH=src python -m lerobot.scripts.export_attention_routing /path/to/evaluation_directory
```

旧 NPZ 只有相机切片，无法恢复其余 key 的概率；必须重新推理，无需重新训练。
更改 layer、checkpoint、配置或采集代码后使用新的 `BASE_OUTPUT`。

第一份分析应交付：

1. 全组总量表与沿 episode 时间的堆叠图，回答 98.76% 主要属于哪一组。
2. head × group、pass × group 热图，检查平均是否掩盖少量视觉 head。
3. 各组每 token 强度、top key 频率与熵，检查高质量特征读取和固定位置聚集的候选现象。
4. 按接近/抓取/搬运/放置阶段比较；阶段由视频/轨迹标注，不从 attention 自己推断。
5. 分开报告成功/失败、task/suite。先汇总 episode，再汇总 task/suite，
   避免长 episode 因 prediction 多而主导结果；不把逐帧样本当成独立试验做显著性检验。

## 3. 依据结果决定下一步（设计，尚未实现干预）

| 观察结果 | 可检验的解释 | 下一步 |
| --- | --- | --- |
| VLM prompt 占比高且少数 prompt 集中 | prompt 可能承载上下文，也可能只是高注意力的固定位置 | 检查同批输入的 prompt→图像/文本；对 prompt 做替换与激活干预 |
| action key 占比高 | 最后层可能在整合 chunk 内动作；视觉可能已在前层进入 | 采集专家浅/中/末层，随后细分同位置/其他动作位置及不同 query |
| language/state 高 | 状态或任务条件可能主导局部动作更新 | 可靠拆分 task/state/special，再做分别替换的配对测试 |
| 少数 head 视觉 mass 高 | 全 head 平均可能掩盖分工 | 对这些 head 单独检查时序与空间图，再与匹配的对照 head 干预比较 |
| 同一 key 持续高，但干预效应小 | 可能是高 attention、低输出影响的聚集位置 | 看组的 value/output 投影贡献及干预后的动作变化；不可仅按位置称为 attention sink |

先取浅/中/末三层（18 层模型可选 0/8/17；以实际层数为准），然后在变化明显处补层。
用相同输入比较 action→各组与 VLM prompt→各组，可检查视觉信息是否可能经 prompt 间接进入动作。
已实现的单层采集器可以分别运行：

```bash
ATTENTION_LAYER=8 BASE_OUTPUT=/root/autodl-tmp/eval/prompt-learning-routing-l8 \
  bash run_eval_libero_attention_routing.sh libero_10 0

ATTENTION_RECORD_ROUTING=1 \
BASE_OUTPUT=/root/autodl-tmp/eval/prompt-learning-vlm-routing \
  bash run_eval_libero_vlm_prompt_attention.sh libero_10 0
```

保留原来的 checkpoint/font 等环境设置。分别运行时核对相同环境 seed、轨迹和输入；
独立闭环 rollout 若发生分叉，不能直接逐帧比较。更严格的下一步是保存固定 observation，
在同一模型和同一初始 action noise 上做离线重放，并恢复每次比较的 RNG 状态。
这部分需要新增固定输入重放工具，当前采集器没有保存完整重放状态。

最后层 image key 对应的是经过 VLM 上下文处理的视觉位置，不是原始像素分割标记。
某层 action→prompt 的 mass 与 prompt→image 的 mass 不能直接相乘当作精确视觉贡献；
残差、MLP、value 和跨层交互均未纳入这两个数。
可把跨层 rollout 作为辅助描述，但不作为因果结论。

## 4. 用干预验证作用，而不只观察权重（设计）

先做固定 observation、固定初始 noise 的开环配对比较，再做相同环境种子的闭环评估。
开环记录完整 action chunk 与实际执行前缀的差异，先在归一化动作空间比较，
并分别报告平移/旋转/夹爪，避免不同物理单位直接混合成一个范数。
闭环报告成功率差、失败阶段、轨迹偏移和配对置信区间。

- **输入 prompt 干预**：学习后的 prompt 与保存的初始化 prompt/零向量/匹配范数随机 prompt
  比较，保持长度、位置和 mask 相同，多次随机重复。它检验该权重是否敏感，
  但零化会改变分布，不能单独证明视觉信息经 prompt 传递。
- **视觉经 prompt 中介的检验**：对同一输入扰动目标图像，保存 clean/corrupt 的 prompt
  hidden state 或对应层 prefix K/V；在固定其他分支与 noise 时替换相应 prompt 激活，
  观察动作是否朝 clean 输出恢复。明确干预层、缓存重算边界与被保持不变的分支。
  VLM 输入 embedding 相同不等于其上下文化 hidden state 相同。
- **视觉对照**：目标物区域遮挡 vs 同面积背景遮挡，主相机 vs 腕部相机，
  使用多种填充值/对照区域；整幅置零只作为粗压力测试，不能独自定位物体依据。
- **head/组贡献**：先记录 `sum(A_group * V_group)` 经 output projection 后的向量，
  再用输出替换/消融检验；向量范数只能描述大小，抵消与后续网络仍影响最终动作。

不要把同一 checkpoint 的推理期破坏，等同于“从头训练一个没有 prompt 的模型”。
要解释 vlm_prompt_only 的相对优势，还需受控训练对照：相同基础权重、数据、优化预算、
checkpoint 选择规则与评估 seeds，比较无 prompt、仅 VLM、仅 action、双 prompt。
同时记录可训练参数量、prompt 数量与训练 seed；已有对照若匹配可先复用，否则标明混杂因素。
最先验证明确假设，再扩大到四套 suite、更多 episode 和多个训练 seed，避免一次运行所有组合。

方法参考（不能替代对本模型的验证）：
[Jain & Wallace, 2019](https://aclanthology.org/N19-1357/) 讨论 attention 作为解释的局限；
[Abnar & Zuidema, 2020](https://aclanthology.org/2020.acl-main.385/) 讨论跨层 attention flow/rollout。
