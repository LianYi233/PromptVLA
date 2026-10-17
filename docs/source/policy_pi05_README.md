# π₀.₅ (pi05)

This repository contains the Hugging Face port of **π₀.₅**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model with open-world generalization**.

---

## Model Overview

| Feature              | π₀                                                     | π₀.₅                                      |
| -------------------- | ------------------------------------------------------ | ----------------------------------------- |
| Time Conditioning    | Concatenates time with actions via `action_time_mlp_*` | Uses `time_mlp_*` for AdaRMS conditioning |
| AdaRMS               | Not used                                               | Used in action expert                     |
| Tokenizer Length     | 48 tokens                                              | 200 tokens                                |
| Discrete State Input | False (Uses `state_proj` layer)                        | True                                      |
| Parameter Count      | Higher (includes state embedding)                      | Lower (no state embedding)                |

---

## Training Defaults

This repository's variant uses
`[VLM observation/language tokens] [VLM prompt tokens] [expert prompt tokens] [action tokens]`.
There are 16 trainable embeddings on each side by default (`num_vlm_prompt_tokens=16`,
`num_prompt_tokens=16`, `prompt_init_std=0.02`), using their respective backbone widths.
With the default VLM/expert variants, the two banks have **32,768 + 16,384 = 49,152 trainable
parameters**. Observation/language tokens attend only to their own block. VLM prompts attend to
observation/language tokens and all VLM prompts; expert prompts attend to all expert prompts and
action tokens. Both prompt banks retain self-attention and bidirectional attention within their
own bank, but neither attends directly to the other prompt bank. Expert prompts do not attend
directly to observation/language tokens. Actions attend to all four blocks, with bidirectional
attention across the action chunk.

The entire VLM, action expert, action projections, and timestep MLP stay frozen in every phase.
`freeze_vision_encoder=true` and `train_expert_only=true` remain enforced legacy flags. The training
curriculum is unchanged: action-only inpainting updates expert prompts; the observation-conditioned
bridge and formal flow training update both prompt banks. In action-only inpainting, expert prompts
still attend to their own bank and the action block while the VLM prefix is absent. At inference,
VLM prompts remain independent of actions and are cached with the VLM prefix. Expert prompts read
the current noisy action tokens and are recomputed at every denoising step; the action chunk length
is unchanged.
With `cabo_enabled=false`, either prompt count may be `0` for an ablation.
Inpainting requires expert prompts, and training requires a nonempty bank. Both counts may be `0`
for inference with CABO disabled.

Fresh configurations use a unified AdamW learning rate of `2.5e-5` for both prompt banks,
followed by cosine decay to a `1e-5` floor. Global gradient clipping is disabled
(`optimizer_grad_clip_norm=0.0`). VLM-relative gradient clipping (`clip_action_head_by_vlm`) must
remain `false` for flow training.

CABO is enabled by default (`cabo_enabled=true`, `cabo_prompt_update_ratio=2.0`) and requires both
prompt banks. It uses named AdamW groups `vlm_prompt` and `action_prompt`. In the bridge and formal
flow phases, it limits the expert prompt's relative learning update to at most half of the VLM
prompt's: `r_action <= r_vlm / 2`, where `r = ||AdamW learning delta||₂ / ||prompt parameters||₂`
excludes decoupled weight decay. Only expert prompt updates above the cap are attenuated; VLM
updates are unchanged. Action-only inpainting bypasses CABO and updates expert prompts normally,
because its forward path omits the VLM. The three phases and learning-rate scheduler are unchanged.
Use a larger `cabo_prompt_update_ratio` for a lower expert prompt cap, or `cabo_enabled=false` to
disable CABO. The ratio accepts values of at least `1.0`. CABO requires
`use_policy_training_preset=true` so its named groups are constructed correctly.

The old expert/projection update ratios and CABO EMA, warmup, and floor settings remain loadable
for compatibility but do not affect prompt CABO, which uses the current VLM prompt update.

Loading a base or older full-model checkpoint initializes missing prompt banks while preserving
existing prompt weights. Loading a checkpoint saved by this variant preserves both learned banks.
To migrate an older training run, initialize a new run from its model weights: the old optimizer
layout differs after adding VLM prompts and freezing the action path. The former single-group
prompt optimizer also differs from CABO's two named prompt groups. See the
[training guide](./pi05.mdx) for the curriculum and checkpoint details.
Saved CABO settings are preserved. When loading a checkpoint that stored CABO disabled or ratio
`1.0`, pass `--policy.cabo_enabled=true --policy.cabo_prompt_update_ratio=2.0` explicitly.

PEFT defaults to saving only the two prompt banks through `modules_to_save`, with no backbone LoRA
adapters. Any custom backbone adapters remain frozen. Legacy adapters must save every enabled bank:
set `num_vlm_prompt_tokens=0` for an expert-prompt-only adapter, or both counts to `0` for inference
with an adapter containing neither bank; set `cabo_enabled=false` in either case. Initialize from a
full-model checkpoint to add missing banks.

## Training performance

The defaults `attention_implementation="sdpa"` and `separate_frozen_observations=true` enable
PyTorch SDPA and separate the frozen observation/language computation in the bridge and Stage 2.
Each layer computes those tokens under `no_grad` and supplies constant keys and values to the
prompt/action computation under the same visibility mask. Both prompt banks retain their gradients.
With gradient checkpointing enabled, only the prompt/action computation is checkpointed once per
layer; the frozen observation branch is not recomputed during backward. Cached inference uses the
configured attention backend too.

The default precision remains full FP32 (`dtype="float32"`); BF16 is opt-in with
`--policy.dtype=bfloat16`, and `compile_model` remains `false` by default. For an eager comparison,
use `--policy.attention_implementation=eager --policy.separate_frozen_observations=false` with the
same precision and batch size. Throughput depends on the hardware and sequence lengths.

---

## Relative Actions

π₀.₅ supports training with **relative actions**, where the model learns relative offsets
from the current robot state instead of absolute joint positions. This mirrors the
relative-action transform in OpenPI (`DeltaActions`) and can improve performance.

### How it works

1. **During preprocessing**, absolute actions are converted to relative offsets:
   `relative = action - state` (for selected joints).
2. The relative actions are normalized using statistics computed from the relative distribution.
3. **During postprocessing**, predicted relative actions are converted back to absolute:
   `absolute = relative + state`.

Joints listed in `relative_exclude_joints` (e.g., gripper) are kept absolute.

### Configuration

| Parameter                 | Type        | Default       | Description                                                      |
| ------------------------- | ----------- | ------------- | ---------------------------------------------------------------- |
| `use_relative_actions`    | `bool`      | `False`       | Enable relative-action training                                  |
| `relative_exclude_joints` | `list[str]` | `["gripper"]` | Joint names to keep absolute (matched by substring)              |
| `action_feature_names`    | `list[str]` | `None`        | Auto-populated from dataset metadata at runtime by `make_policy` |

### Training example

```bash
uv run lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=your_org/your_dataset \
  --policy.num_vlm_prompt_tokens=16 \
  --policy.num_prompt_tokens=16 \
  --policy.cabo_enabled=true \
  --policy.cabo_prompt_update_ratio=2.0 \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]'
```

When `use_relative_actions=true`, the training script automatically:

- Computes relative action statistics from the dataset (sampled chunk-level relative actions)
- Replaces the standard action stats with relative stats for normalization
- Broadcasts these stats across all ranks in distributed training

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
