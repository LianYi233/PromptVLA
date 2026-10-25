#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.cabo import (
    CABO_ACTION_PROMPT_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_PROMPT_GROUP,
    OptimizerStepControl,
    adamw_group_relative_update_moments,
    get_named_param_group,
    relative_update_rate,
    require_adamw,
    validate_adamw_param_group,
)
from lerobot.policies.pi05.attention_pi05 import pi05_attention_forward
from lerobot.policies.pi05.configuration_pi05 import (
    DEFAULT_IMAGE_SIZE,
    PI05Config,
    PI05TrainingObjective,
    PI05TrainingStage,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

_PI05_LOSS_VALID_STEPS_KEY = "pi05_loss_valid_steps"


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _unique_parameters_with_grad(modules: list[nn.Module]) -> list[nn.Parameter]:
    """Collect unique, trainable parameters that received a gradient."""
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen or not parameter.requires_grad or parameter.grad is None:
                continue
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def _unique_trainable_parameters(modules: list[nn.Module]) -> list[nn.Parameter]:
    """Collect unique trainable parameters without requiring an existing gradient."""
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen or not parameter.requires_grad:
                continue
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def _gradient_l2_norm_and_numel(parameters: list[nn.Parameter]) -> tuple[Tensor | None, int]:
    """Return the group L2 gradient norm and the number of gradient elements."""
    gradient_norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2, dtype=torch.float32)
        for parameter in parameters
    ]
    if not gradient_norms:
        return None, 0
    return torch.stack(gradient_norms).norm(2), sum(parameter.grad.numel() for parameter in parameters)


def _valid_action_steps(batch: dict[str, Tensor]) -> Tensor:
    """Return the action timesteps that contain real, non-padding targets."""
    actions = batch[ACTION]
    if actions.ndim < 2:
        raise ValueError(f"{ACTION} must include batch and horizon dimensions, got {tuple(actions.shape)}")

    batch_size, action_horizon = actions.shape[:2]
    padding_key = f"{ACTION}_is_pad"
    if padding_key not in batch:
        return torch.ones((batch_size, action_horizon), dtype=torch.bool, device=actions.device)
    action_is_pad = batch[padding_key]
    if not isinstance(action_is_pad, Tensor):
        raise TypeError(f"{padding_key} must be a tensor, got {type(action_is_pad).__name__}")
    expected_shape = (batch_size, action_horizon)
    if action_is_pad.shape != expected_shape:
        raise ValueError(
            f"{ACTION}_is_pad must have shape {expected_shape}, got {tuple(action_is_pad.shape)}"
        )

    return ~action_is_pad.to(device=actions.device, dtype=torch.bool)


def _reduce_action_losses(
    losses: Tensor,
    valid_steps: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce elementwise action losses while excluding padded episode tails."""
    if losses.ndim != 3:
        raise ValueError(
            f"PI0.5 losses must have shape (batch, horizon, action_dim), got {tuple(losses.shape)}"
        )
    if valid_steps.shape != losses.shape[:2]:
        raise ValueError(
            "PI0.5 valid-step mask must match the loss batch and horizon dimensions; "
            f"got loss shape={tuple(losses.shape)}, mask shape={tuple(valid_steps.shape)}"
        )

    valid_elements = valid_steps.to(device=losses.device, dtype=torch.bool).unsqueeze(-1).expand_as(losses)
    masked_losses = torch.where(valid_elements, losses, torch.zeros_like(losses))

    valid_per_sample = valid_elements.sum(dim=(1, 2))
    per_sample_loss = masked_losses.sum(dim=(1, 2)) / valid_per_sample.clamp_min(1)
    valid_per_dim = valid_elements.sum(dim=(0, 1))
    loss_per_dim = masked_losses.sum(dim=(0, 1)) / valid_per_dim.clamp_min(1)
    scalar_loss = masked_losses.sum() / valid_elements.sum().clamp_min(1)
    return scalar_loss, per_sample_loss, loss_per_dim


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_pi05_att_2d_masks(
    pad_masks: Tensor,
    *,
    prefix_length: int,
    num_vlm_prompt_tokens: int,
    num_prompt_tokens: int,
    query_offset: int = 0,
) -> Tensor:
    """Mask [observations, VLM prompts, expert prompts, actions] by direct token visibility.

    Observations read only observations. VLM prompts read observations and VLM prompts;
    expert prompts read actions and expert prompts. Actions read every valid token.
    ``query_offset=prefix_length`` builds only suffix query rows for cached denoising.
    """
    positions = torch.arange(pad_masks.shape[1], device=pad_masks.device)
    queries = positions[query_offset:, None]
    keys = positions[None, :]
    observation_end = prefix_length - num_vlm_prompt_tokens
    action_start = prefix_length + num_prompt_tokens

    action_queries = queries >= action_start
    allowed = torch.where(
        queries < observation_end,
        keys < observation_end,
        torch.where(queries < prefix_length, keys < prefix_length, action_queries | (keys >= prefix_length)),
    )
    return allowed[None, :, :] & pad_masks[:, query_offset:, None] & pad_masks[:, None, :]


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


def _layer_qkv(layer, hidden_states, condition):
    normalized, gate = layernorm_forward(layer.input_layernorm, hidden_states, condition)
    head_dim = layer.self_attn.head_dim
    states = []
    for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
        projected = projection(normalized)
        states.append(
            projected.view(*normalized.shape[:-1], projected.shape[-1] // head_dim, head_dim).transpose(1, 2)
        )
    query, key, value = states
    return query, key, value, gate


def _layer_output(layer, hidden_states, attention_output, gate, condition):
    attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
    residual = _gated_residual(hidden_states, layer.self_attn.o_proj(attention_output), gate)
    normalized, gate = layernorm_forward(layer.post_attention_layernorm, residual, condition)
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        normalized = normalized.to(dtype=torch.bfloat16)
    return _gated_residual(residual, layer.mlp(normalized), gate)


@torch.no_grad()
def compute_layer_frozen_observations(
    layer_idx, observations, attention_mask, position_ids, paligemma, attention_implementation="sdpa"
):
    """Advance frozen observations once and expose their constant per-layer keys and values."""
    language_model = paligemma.model.language_model
    layer = language_model.layers[layer_idx]
    query, key, value, gate = _layer_qkv(layer, observations, None)
    cos, sin = language_model.rotary_emb(query, position_ids)
    query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    output, _ = pi05_attention_forward(
        layer.self_attn,
        query,
        key,
        value,
        attention_mask,
        layer.self_attn.scaling,
        implementation=attention_implementation,
    )
    output = output.reshape(observations.shape[0], observations.shape[1], query.shape[1] * query.shape[-1])
    return _layer_output(layer, observations, output, gate, None), (key, value)


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
    attention_implementation="sdpa",
    frozen_observation_kv=None,
):
    """Advance trainable token paths, optionally attending to constant observation keys/values."""
    models = [paligemma.model.language_model, gemma_expert.model]
    projected = [
        _layer_qkv(models[i].layers[layer_idx], hidden, adarms_cond[i])
        for i, hidden in enumerate(inputs_embeds)
    ]
    query = torch.cat([item[0] for item in projected], dim=2)
    key = torch.cat([item[1] for item in projected], dim=2)
    value = torch.cat([item[2] for item in projected], dim=2)
    cos, sin = models[0].rotary_emb(query, position_ids)
    query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    if frozen_observation_kv is not None:
        key = torch.cat([frozen_observation_kv[0], key], dim=2)
        value = torch.cat([frozen_observation_kv[1], value], dim=2)
    attention = models[0].layers[layer_idx].self_attn
    output, _ = pi05_attention_forward(
        attention,
        query,
        key,
        value,
        attention_mask,
        attention.scaling,
        implementation=attention_implementation,
    )
    output = output.reshape(query.shape[0], query.shape[2], query.shape[1] * query.shape[-1])
    outputs = []
    start = 0
    for i, hidden in enumerate(inputs_embeds):
        end = start + hidden.shape[1]
        outputs.append(
            _layer_output(
                models[i].layers[layer_idx], hidden, output[:, start:end], projected[i][3], adarms_cond[i]
            )
        )
        start = end
    return outputs


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        attention_implementation: str = "sdpa",
        separate_frozen_observations: bool = True,
        num_vlm_prompt_tokens: int = 0,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.separate_frozen_observations = separate_frozen_observations
        self.num_vlm_prompt_tokens = num_vlm_prompt_tokens

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.text_config._attn_implementation = attention_implementation
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
            attn_implementation=attention_implementation,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.paligemma.model.language_model.config._attn_implementation = attention_implementation
        self.gemma_expert.model.config._attn_implementation = attention_implementation

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()
        return self

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Observations cannot read prompts/actions under the PI05 mask. Their entire frozen
            # path can run without autograd, while their per-layer KV remains available to prompts
            # and actions. Keep this outside checkpointing so backward never recomputes observations.
            observation_length = 0
            observations = None
            if self.separate_frozen_observations and self.train_expert_only and attention_mask is not None:
                observation_length = inputs_embeds[0].shape[1] - self.num_vlm_prompt_tokens
                if observation_length > 0:
                    observations = inputs_embeds[0][:, :observation_length].detach()
                    inputs_embeds = [inputs_embeds[0][:, observation_length:], inputs_embeds[1]]
            observation_mask = (
                attention_mask[:, :, :observation_length, :observation_length]
                if attention_mask is not None
                else None
            )
            train_attention_mask = (
                attention_mask[:, :, observation_length:, :] if attention_mask is not None else None
            )
            observation_positions = position_ids[:, :observation_length]
            train_positions = position_ids[:, observation_length:]

            for layer_idx in range(num_layers):
                frozen_observation_kv = None
                if observations is not None:
                    observations, frozen_observation_kv = compute_layer_frozen_observations(
                        layer_idx,
                        observations,
                        observation_mask,
                        observation_positions,
                        self.paligemma,
                        self.attention_implementation,
                    )
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        train_attention_mask,
                        train_positions,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        attention_implementation=self.attention_implementation,
                        frozen_observation_kv=frozen_observation_kv,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        train_attention_mask,
                        train_positions,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        attention_implementation=self.attention_implementation,
                        frozen_observation_kv=frozen_observation_kv,
                    )

            prefix_output, _ = layernorm_forward(models[0].norm, inputs_embeds[0], adarms_cond[0])
            suffix_output, _ = layernorm_forward(models[1].norm, inputs_embeds[1], adarms_cond[1])
            if observations is not None:
                with torch.no_grad():
                    observations, _ = layernorm_forward(models[0].norm, observations, None)
                prefix_output = torch.cat([observations, prefix_output], dim=1)
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=True,
            attention_implementation=config.attention_implementation,
            separate_frozen_observations=config.separate_frozen_observations,
            num_vlm_prompt_tokens=config.num_vlm_prompt_tokens,
        )

        self.vlm_prompt_tokens = nn.Embedding(config.num_vlm_prompt_tokens, paligemma_config.width)
        nn.init.normal_(self.vlm_prompt_tokens.weight, std=config.prompt_init_std)
        self.prompt_tokens = nn.Embedding(config.num_prompt_tokens, action_expert_config.width)
        nn.init.normal_(self.prompt_tokens.weight, std=config.prompt_init_std)

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Persist the former inpainting role signal so Stage-1 and Stage-2 state dicts remain identical.
        # It stays zero and frozen for checkpoint compatibility; visible actions remain clean while
        # masked actions follow the sampled flow path.
        self.inpainting_visible_action_embedding = nn.Embedding(1, action_expert_config.width)
        nn.init.zeros_(self.inpainting_visible_action_embedding.weight)
        self.inpainting_visible_action_embedding.requires_grad_(False)
        self._freeze_vlm()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # FP32 also requires full-precision matmuls and vision convolutions. The trainer may
        # have enabled TF32 before constructing the model; override it before compiling.
        if config.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
        elif config.compile_model:
            torch.set_float32_matmul_precision("high")

        # Compile model if requested
        if config.compile_model:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _freeze_vlm(self) -> None:
        """Compatibility entry point for enforcing the configured trainable partition."""
        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        """Freeze backbones, preserving active prompts and opt-in action projections.

        Preserve PEFT's active modules_to_save selection: its original prompt copies must stay
        frozen. Autograd remains enabled through both backbones to reach the input prompts.
        """
        trainable_ids = set()
        trainable_modules = [self.vlm_prompt_tokens, self.prompt_tokens]
        if getattr(self.config, "train_action_projections", False):
            trainable_modules.extend([self.action_in_proj, self.action_out_proj])
        for module in trainable_modules:
            modules_to_save = getattr(module, "modules_to_save", None)
            if modules_to_save is None:
                active_modules = [module]
            else:
                # Only PEFT's selected copies are used by forward. Never enable original or
                # inactive copies, even after an external requires_grad_(True) on the policy.
                active_modules = [
                    modules_to_save[name]
                    for name in module.active_adapters
                    if name in modules_to_save and not module.disable_adapters
                ]
            trainable_ids.update(
                id(parameter)
                for active_module in active_modules
                for parameter in active_module.parameters()
                if parameter.requires_grad and parameter.numel() > 0
            )
        for parameter in self.parameters():
            if id(parameter) not in trainable_ids:
                parameter.requires_grad_(False)
                parameter.grad = None
        self.paligemma_with_expert.paligemma.eval()
        self.paligemma_with_expert.gemma_expert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_vlm()
        return self

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        # These helpers wrap frozen modules. Without a differentiable input there is no
        # backward graph to save; checkpointing the frozen vision encoder also interferes
        # with torch.compile through Transformers' output-capturing context.
        if (
            self.gradient_checkpointing_enabled
            and self.training
            and any(isinstance(arg, Tensor) and arg.requires_grad for arg in args)
        ):
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed [images, language, VLM prompts] without detaching prompt gradients."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            return self.paligemma_with_expert.embed_language_tokens(tokens)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if self.config.num_vlm_prompt_tokens:
            prompt_ids = torch.arange(self.config.num_vlm_prompt_tokens, device=lang_emb.device)
            prompt_embs = self.vlm_prompt_tokens(prompt_ids).to(dtype=lang_emb.dtype)
            embs.append(prompt_embs.unsqueeze(0).expand(lang_emb.shape[0], -1, -1))
            pad_masks.append(
                torch.ones(
                    lang_emb.shape[0],
                    self.config.num_vlm_prompt_tokens,
                    dtype=torch.bool,
                    device=lang_emb.device,
                )
            )
            # Retain layout markers in the embedding API. make_pi05_att_2d_masks defines
            # visibility: VLM prompts read observations and their own prompt bank.
            att_masks += [1] + [0] * (self.config.num_vlm_prompt_tokens - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed [prompt, action] blocks with shared timestep conditioning for the expert."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        bsize, action_time_dim = action_time_emb.shape[:2]
        if self.config.num_prompt_tokens:
            # Use the embedding forward so PEFT modules_to_save also routes to the active adapter.
            prompt_ids = torch.arange(self.config.num_prompt_tokens, device=action_emb.device)
            prompt_embs = self.prompt_tokens(prompt_ids).to(dtype=action_emb.dtype)
            embs.append(prompt_embs.unsqueeze(0).expand(bsize, -1, -1))
            pad_masks.append(
                torch.ones(bsize, self.config.num_prompt_tokens, dtype=torch.bool, device=action_emb.device)
            )
            # Expert prompts read actions and their own prompt bank under make_pi05_att_2d_masks.
            # The VLM prefix cannot read this block and remains cacheable across denoising steps.
            att_masks += [1] + [0] * (self.config.num_prompt_tokens - 1)

        embs.append(action_time_emb)
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Actions see the VLM, prompts, and the entire bidirectional action block.
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _validate_inpainting_inputs(
        self,
        actions: Tensor,
        action_is_pad: Tensor | None,
        inpainting_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if action_is_pad is None:
            raise ValueError("action_is_pad is required for action-only flow pretraining")
        if actions.ndim != 3 or actions.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"actions must have shape [batch, {self.config.chunk_size}, action_dim], "
                f"got {tuple(actions.shape)}"
            )
        expected_mask_shape = actions.shape[:2]
        if action_is_pad.shape != expected_mask_shape:
            raise ValueError(
                f"action_is_pad must have shape {expected_mask_shape}, got {tuple(action_is_pad.shape)}"
            )
        action_is_pad = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if inpainting_mask is not None:
            if inpainting_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"inpainting_mask must have shape {expected_mask_shape}, "
                    f"got {tuple(inpainting_mask.shape)}"
                )
            inpainting_mask = inpainting_mask.to(device=actions.device, dtype=torch.bool)
            masks_padding = inpainting_mask & action_is_pad
            if not torch.compiler.is_compiling() and masks_padding.any():
                invalid_samples = masks_padding.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    "inpainting_mask cannot select padded action positions; "
                    f"invalid batch indices {invalid_samples}"
                )
        return action_is_pad, inpainting_mask

    def sample_inpainting_mask(self, action_is_pad: Tensor) -> Tensor:
        """Uniformly mask up to ``next_action_masked_steps`` valid time positions per sample."""
        if action_is_pad.ndim != 2 or action_is_pad.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"action_is_pad must have shape [batch, {self.config.chunk_size}], "
                f"got {tuple(action_is_pad.shape)}"
            )
        valid = ~action_is_pad.to(dtype=torch.bool)
        if self.config.next_action_full_mask_probability == 1.0:
            return valid

        random_scores = torch.rand(valid.shape, device=valid.device)
        random_scores = random_scores.masked_fill(~valid, torch.inf)
        ranks = random_scores.argsort(dim=1).argsort(dim=1)
        num_to_mask = valid.sum(dim=1).clamp(max=self.config.next_action_masked_steps)
        inpainting_mask = valid & (ranks < num_to_mask[:, None])
        if self.config.next_action_full_mask_probability > 0.0:
            full_mask = (
                torch.rand(valid.shape[0], device=valid.device)
                < self.config.next_action_full_mask_probability
            )
            inpainting_mask = torch.where(full_mask[:, None], valid, inpainting_mask)
        return inpainting_mask

    def predict_inpainting_velocity(
        self,
        x_t: Tensor,
        time: Tensor,
        action_is_pad: Tensor,
        inpainting_mask: Tensor,
    ) -> Tensor:
        """Predict the masked-action flow without a VLM prefix using the formal flow modules."""
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(x_t, time)
        visible = (~action_is_pad.to(device=x_t.device, dtype=torch.bool)) & (~inpainting_mask)
        num_prompts = self.config.num_prompt_tokens
        action_embs = suffix_embs[:, num_prompts:] + visible.unsqueeze(-1).to(suffix_embs.dtype) * (
            self.inpainting_visible_action_embedding.weight[0].to(suffix_embs.dtype)
        )
        suffix_embs = torch.cat([suffix_embs[:, :num_prompts], action_embs], dim=1)
        suffix_pad_masks = torch.cat(
            [suffix_pad_masks[:, :num_prompts], ~action_is_pad.to(device=x_t.device, dtype=torch.bool)], dim=1
        )
        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)
        att_2d_masks = make_pi05_att_2d_masks(
            suffix_pad_masks,
            prefix_length=0,
            num_vlm_prompt_tokens=0,
            num_prompt_tokens=num_prompts,
        )
        position_ids = (
            torch.arange(suffix_embs.shape[1], dtype=torch.long, device=x_t.device)
            .unsqueeze(0)
            .expand(x_t.shape[0], -1)
        )
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ).to(dtype=torch.float32)
        suffix_out = suffix_out[:, num_prompts:]
        return self._apply_checkpoint(self.action_out_proj, suffix_out)

    def predict_velocity(self, images, img_masks, tokens, masks, x_t, time) -> Tensor:
        """Predict flow velocity for explicit noisy actions and timesteps."""
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_2d_masks = make_pi05_att_2d_masks(
            pad_masks,
            prefix_length=prefix_pad_masks.shape[1],
            num_vlm_prompt_tokens=self.config.num_vlm_prompt_tokens,
            num_prompt_tokens=self.config.num_prompt_tokens,
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        # The joint transformer checkpoints each trainable layer internally. An outer checkpoint
        # would replay the whole stack and then replay its individual layers again in backward.
        suffix_out = forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    def forward(
        self,
        images=None,
        img_masks=None,
        tokens=None,
        masks=None,
        actions=None,
        noise=None,
        time=None,
        action_is_pad=None,
        inpainting_mask=None,
        objective: PI05TrainingObjective | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if actions is None:
            raise ValueError("actions are required")
        if objective is None:
            objective = (
                PI05TrainingObjective.ACTION_INPAINTING
                if self.config.training_stage == PI05TrainingStage.NEXT_ACTION
                else PI05TrainingObjective.OBSERVATION_FLOW
            )
        if objective == PI05TrainingObjective.ACTION_INPAINTING:
            action_is_pad, inpainting_mask = self._validate_inpainting_inputs(
                actions, action_is_pad, inpainting_mask
            )
            if inpainting_mask is None:
                inpainting_mask = self.sample_inpainting_mask(action_is_pad)
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            valid_elements = (~action_is_pad).unsqueeze(-1)
            safe_actions = torch.where(valid_elements, actions, torch.zeros_like(actions))
            safe_noise = torch.where(valid_elements, noise, torch.zeros_like(noise))
            time_expanded = time[:, None, None]
            noisy_actions = time_expanded * safe_noise + (1 - time_expanded) * safe_actions
            masked_elements = inpainting_mask.unsqueeze(-1)
            # Masked actions follow the flow path while visible actions stay clean and provide
            # bidirectional context. Padding is sanitized above and never contributes to the loss.
            x_t = torch.where(masked_elements, noisy_actions, safe_actions)
            u_t = safe_noise - safe_actions
            v_t = self.predict_inpainting_velocity(x_t, time, action_is_pad, inpainting_mask)
            losses = F.mse_loss(u_t, v_t, reduction="none")
            return torch.where(masked_elements, losses, torch.zeros_like(losses))

        if objective != PI05TrainingObjective.OBSERVATION_FLOW:
            raise ValueError(f"Unsupported PI0.5 training objective: {objective!r}")

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        v_t = self.predict_velocity(images, img_masks, tokens, masks, x_t, time)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        return_prefix_features: bool = False,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Do a full inference forward and compute the action."""
        if self.config.training_stage == "next_action":
            raise RuntimeError(
                "Action inference is unavailable for a frozen-VLM Stage-1 checkpoint; "
                "load it with training_stage='flow' first"
            )
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_pi05_att_2d_masks(
            prefix_pad_masks,
            prefix_length=prefix_pad_masks.shape[1],
            num_vlm_prompt_tokens=self.config.num_vlm_prompt_tokens,
            num_prompt_tokens=0,
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        prefix_outputs, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_output = prefix_outputs[0]
        prefix_mask = prefix_pad_masks.to(prefix_output.dtype).unsqueeze(-1)
        prefix_features = (prefix_output * prefix_mask).sum(dim=1) / prefix_mask.sum(dim=1).clamp_min(1.0)
        prefix_features = prefix_features.to(dtype=torch.float32)

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if return_prefix_features:
            return x_t, prefix_features
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(x_t, timestep)
        prefix_len = prefix_pad_masks.shape[1]
        full_att_2d_masks = make_pi05_att_2d_masks(
            torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1),
            prefix_length=prefix_len,
            num_vlm_prompt_tokens=self.config.num_vlm_prompt_tokens,
            num_prompt_tokens=self.config.num_prompt_tokens,
            query_offset=prefix_len,
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"
    _LEGACY_NEXT_ACTION_CHECKPOINT_KEYS = frozenset(
        {
            "model.next_action_query",
            "model.next_action_out_proj.weight",
            "model.next_action_out_proj.bias",
        }
    )
    _INPAINTING_ROLE_KEY = "model.inpainting_visible_action_embedding.weight"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)
        # Runtime-only curriculum state. The trainer sets this from its checkpoint-aware global
        # update counter before each forward. Keeping it out of the state dict preserves strict
        # Stage-1 -> Stage-2 checkpoint compatibility.
        self._stage1_bridge_active = False

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def set_training_progress(self, *, step: int, total_steps: int) -> None:
        """Select the Stage-1 objective for the next update without changing model state."""
        self._stage1_bridge_active = False
        if self.config.training_stage != PI05TrainingStage.NEXT_ACTION:
            return
        if total_steps <= 0:
            raise ValueError(f"total_steps must be greater than 0, got {total_steps}")
        if step < 0 or step >= total_steps:
            raise ValueError(f"step must be in [0, {total_steps}), got {step}")
        bridge_steps = self.config.next_action_bridge_steps
        if bridge_steps > total_steps:
            raise ValueError(
                "next_action_bridge_steps cannot exceed the Stage-1 training steps, "
                f"got {bridge_steps} > {total_steps}"
            )
        self._stage1_bridge_active = bridge_steps > 0 and step >= total_steps - bridge_steps

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load PI05 model.safetensors from {pretrained_name_or_path!s}"
                ) from e

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(
                original_state_dict, model.config
            )  # ✓ Remapped 812 state dict keys

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            legacy_keys = cls._LEGACY_NEXT_ACTION_CHECKPOINT_KEYS
            present_legacy_keys = legacy_keys.intersection(remapped_state_dict)
            if present_legacy_keys:
                if present_legacy_keys != legacy_keys:
                    raise RuntimeError(
                        "Invalid legacy PI0.5 next-action checkpoint: temporary parameters must be "
                        f"all present or all absent; missing={sorted(legacy_keys - present_legacy_keys)}"
                    )
                logging.warning(
                    "Loading a legacy PI0.5 next-action checkpoint: its temporary regression query/head "
                    "are discarded because action-only flow now trains the formal flow head directly."
                )
                remapped_state_dict = {
                    key: value for key, value in remapped_state_dict.items() if key not in legacy_keys
                }

            if cls._INPAINTING_ROLE_KEY not in remapped_state_dict:
                # Official and pre-inpainting PI0.5 checkpoints predate this zero-initialized role
                # embedding. It has no effect on ordinary flow, so adding the model default is exact.
                remapped_state_dict[cls._INPAINTING_ROLE_KEY] = model.state_dict()[cls._INPAINTING_ROLE_KEY]

            for prompt_key in ("model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"):
                if prompt_key not in remapped_state_dict:
                    # Base/older checkpoints may predate either prompt bank. A saved tensor with
                    # a mismatched shape must still fail; never replace an existing learned prompt.
                    remapped_state_dict[prompt_key] = model.state_dict()[prompt_key]
                    logging.info("Initialized %s absent from the PI05 checkpoint", prompt_key)

            # Inpainting and formal flow deliberately use an identical parameter set. Loading stays
            # strict across stages so no action-flow head can be silently dropped or reinitialized.
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
            if strict and (missing_keys or unexpected_keys):
                raise RuntimeError(
                    "PI0.5 checkpoint is incompatible with the requested training stage: "
                    f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
                )
            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PI05 pretrained weights from {pretrained_name_or_path!s}: {e}"
            ) from e

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def _vlm_modules(self) -> list[nn.Module]:
        return [self.model.paligemma_with_expert.paligemma]

    def _prompt_modules(self) -> list[nn.Module]:
        return [self.model.vlm_prompt_tokens, self.model.prompt_tokens]

    def _action_expert_modules(self) -> list[nn.Module]:
        return [self.model.paligemma_with_expert.gemma_expert]

    def _action_projection_modules(self) -> list[nn.Module]:
        return [
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
            self.model.inpainting_visible_action_embedding,
        ]

    def _action_modules(self) -> list[nn.Module]:
        return [
            self.model.paligemma_with_expert.gemma_expert,
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
            self.model.inpainting_visible_action_embedding,
        ]

    def _is_cabo_active(self) -> bool:
        # The fallback keeps policy hook unit-test stand-ins compatible while the real PI05Config
        # exposes the ``cabo_active`` property.
        cabo_active = getattr(self.config, "cabo_active", None)
        if cabo_active is not None:
            return bool(cabo_active)
        return bool(getattr(self.config, "cabo_enabled", False))

    def _cabo_parameter_groups(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        vlm_parameters = _unique_trainable_parameters([self.model.vlm_prompt_tokens])
        action_parameters = _unique_trainable_parameters([self.model.prompt_tokens])
        expected_groups = {
            CABO_VLM_PROMPT_GROUP: vlm_parameters,
            CABO_ACTION_PROMPT_GROUP: action_parameters,
        }
        group_ids = {
            name: {id(parameter) for parameter in parameters} for name, parameters in expected_groups.items()
        }
        names = list(group_ids)
        for index, name in enumerate(names):
            for other_name in names[index + 1 :]:
                if group_ids[name] & group_ids[other_name]:
                    raise ValueError(f"CABO parameter groups {name!r} and {other_name!r} must be disjoint")

        all_trainable = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        grouped = set().union(*group_ids.values())
        if grouped != all_trainable:
            missing = len(all_trainable - grouped)
            extra = len(grouped - all_trainable)
            raise ValueError(
                "CABO parameter grouping must cover every trainable PI0.5 parameter exactly once; "
                f"missing={missing}, extra={extra}"
            )
        empty_groups = [name for name, parameters in expected_groups.items() if not parameters]
        if empty_groups:
            raise ValueError(
                f"CABO requires non-empty trainable VLM prompt and action prompt groups; empty={empty_groups}"
            )
        return vlm_parameters, action_parameters

    def get_optim_params(self):
        self.model._freeze_vlm()
        if getattr(self.config, "vlm_prompt_only", False):
            parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
            prompt = self.model.vlm_prompt_tokens.weight
            if len(parameters) != 1 or parameters[0] is not prompt or prompt.numel() == 0:
                raise ValueError("vlm_prompt_only must optimize exactly model.vlm_prompt_tokens.weight")
            return parameters
        if self.config.training_stage == PI05TrainingStage.NEXT_ACTION and self.config.num_prompt_tokens == 0:
            raise ValueError("Action-only prompt training requires num_prompt_tokens > 0")
        if not PI05Policy._is_cabo_active(self):
            parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
            if not parameters:
                raise ValueError("PI05 training requires trainable prompts or explicit action projections")
            return parameters
        vlm_parameters, action_parameters = self._cabo_parameter_groups()
        return [
            {"params": vlm_parameters, CABO_GROUP_NAME: CABO_VLM_PROMPT_GROUP},
            {"params": action_parameters, CABO_GROUP_NAME: CABO_ACTION_PROMPT_GROUP},
        ]

    def validate_optimizer_step_control(self, optimizer) -> None:
        """Validate CABO's optimizer contract before the first training backward pass."""
        if not PI05Policy._is_cabo_active(self):
            return

        adamw = require_adamw(optimizer)
        if len(adamw.param_groups) != 2:
            raise ValueError(
                f"CABO requires exactly two named AdamW parameter groups, got {len(adamw.param_groups)}"
            )
        expected_vlm, expected_action = self._cabo_parameter_groups()
        expected_groups = {
            CABO_VLM_PROMPT_GROUP: {id(parameter) for parameter in expected_vlm},
            CABO_ACTION_PROMPT_GROUP: {id(parameter) for parameter in expected_action},
        }
        for name, expected_ids in expected_groups.items():
            group = get_named_param_group(adamw, name)
            # torch AdamW may automatically select foreach on CUDA when this is left as None. CABO's
            # candidate simulator follows the single-tensor implementation, so make that execution
            # path explicit before the first optimizer step.
            if group.get("foreach") is None:
                group["foreach"] = False
            if group.get("fused") is None:
                group["fused"] = False
            validate_adamw_param_group(group)
            actual_parameters = list(group["params"])
            actual_ids = {id(parameter) for parameter in actual_parameters}
            if len(actual_ids) != len(actual_parameters) or actual_ids != expected_ids:
                raise ValueError(
                    f"CABO optimizer group {name!r} does not match PI0.5's parameter partition: "
                    f"expected={len(expected_ids)}, actual={len(actual_ids)}"
                )

        all_optimizer_parameters = [
            parameter for group in adamw.param_groups for parameter in group["params"]
        ]
        all_optimizer_ids = {id(parameter) for parameter in all_optimizer_parameters}
        expected_all_ids = set().union(*expected_groups.values())
        if len(all_optimizer_ids) != len(all_optimizer_parameters) or all_optimizer_ids != expected_all_ids:
            raise ValueError(
                "CABO optimizer parameter groups must cover every trainable PI0.5 parameter exactly once"
            )

    def compute_optimizer_step_control(
        self,
        batch: dict[str, Tensor],
        optimizer,
        accelerator,
    ) -> OptimizerStepControl:
        """Limit action-prompt AdamW learning updates relative to the VLM prompts.

        Action-only inpainting has no VLM reference and keeps its scheduled action-prompt LR.
        The bridge and formal flow compare current, weight-decay-free relative updates, retaining
        the VLM LR and verifying the attenuated action update in the parameter's native dtype.
        """
        if not PI05Policy._is_cabo_active(self):
            return OptimizerStepControl()
        if getattr(self.config, "training_stage", "flow") == PI05TrainingStage.NEXT_ACTION and not getattr(
            self, "_stage1_bridge_active", False
        ):
            return OptimizerStepControl(metrics={"cabo/active": 0.0, "cabo/action_only_bypass": 1.0})

        _ = batch
        adamw = require_adamw(optimizer)
        vlm_group = get_named_param_group(adamw, CABO_VLM_PROMPT_GROUP)
        action_group = get_named_param_group(adamw, CABO_ACTION_PROMPT_GROUP)
        group_moments = []
        active_numels = []
        for group in (vlm_group, action_group):
            update_norm_sq, parameter_norm_sq, active_numel = adamw_group_relative_update_moments(
                group, adamw.state, include_weight_decay=False
            )
            group_moments.extend((update_norm_sq, parameter_norm_sq))
            active_numels.append(active_numel)

        moments = torch.stack(group_moments)
        num_processes = int(getattr(accelerator, "num_processes", 1))
        if num_processes > 1:
            moments = accelerator.reduce(moments, reduction="sum")
        vlm_rate, action_rate = (
            relative_update_rate(float(moments[index].item()), float(moments[index + 1].item()))
            for index in (0, 2)
        )
        ratio = self.config.cabo_prompt_update_ratio
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError("CABO cabo_prompt_update_ratio must be finite and at least 1.0")
        finite = all(math.isfinite(rate) and rate >= 0.0 for rate in (vlm_rate, action_rate))
        update_limit = vlm_rate / ratio if finite else float("nan")
        action_scale = 1.0
        if finite and action_rate > update_limit:
            action_scale = update_limit / action_rate
            if action_scale * action_rate > update_limit:
                action_scale = math.nextafter(action_scale, 0.0)
        scaled_rate = action_rate * action_scale
        verification_passes = 0
        if finite and 0.0 < action_scale < 1.0:
            # Native-dtype rounding need not be linear in LR. All ranks make identical decisions
            # from reduced moments, including when the reference has no gradient on one rank.
            for attempt in range(8):
                candidate_group = {**action_group, "lr": float(action_group["lr"]) * action_scale}
                update_sq, parameter_sq, _ = adamw_group_relative_update_moments(
                    candidate_group, adamw.state, include_weight_decay=False
                )
                checked_moments = torch.stack((update_sq, parameter_sq))
                if num_processes > 1:
                    checked_moments = accelerator.reduce(checked_moments, reduction="sum")
                verification_passes += 1
                scaled_rate = relative_update_rate(
                    float(checked_moments[0].item()), float(checked_moments[1].item())
                )
                if not math.isfinite(scaled_rate):
                    finite = False
                    break
                if scaled_rate <= update_limit:
                    break
                if attempt == 7:
                    action_scale = 0.0
                    scaled_rate = 0.0
                else:
                    action_scale *= 0.5

        metrics = {
            "cabo/active": 1.0,
            "cabo/action_only_bypass": 0.0,
            "cabo/vlm_prompt_relative_update_rate": vlm_rate,
            "cabo/action_prompt_relative_update_rate": action_rate,
            "cabo/prompt_update_ratio": ratio,
            "cabo/action_prompt_relative_update_limit": update_limit,
            "cabo/scaled_action_prompt_relative_update_rate": scaled_rate,
            "cabo/vlm_prompt_scale": 1.0,
            "cabo/action_prompt_scale": action_scale,
            "cabo/candidate_verification_passes": float(verification_passes),
            "cabo/vlm_prompt_active_numel": float(active_numels[0]),
            "cabo/action_prompt_active_numel": float(active_numels[1]),
            "cabo/effective_vlm_prompt_lr": float(vlm_group["lr"]),
            "cabo/effective_action_prompt_lr": float(action_group["lr"]) * action_scale,
            "cabo/update_nonfinite": float(not finite),
        }
        if not finite:
            metrics.update({"optimizer_step/skipped": 1.0, "optimizer_step/nonfinite_cabo_update": 1.0})
            return OptimizerStepControl(metrics=metrics, skip_optimizer_step=True)
        return OptimizerStepControl(group_scales={CABO_ACTION_PROMPT_GROUP: action_scale}, metrics=metrics)

    @torch.no_grad()
    def clip_gradients(self, accelerator=None) -> dict[str, float]:
        """Clip action-side gradients against the VLM's mean gradient magnitude.

        The action side contains the Gemma action expert, action input/output
        projections, and time-conditioning MLPs. The comparison uses the RMS
        gradient per element so the much larger VLM parameter count does not
        dominate the threshold:

            action_l2 / sqrt(action_numel)
                <= ratio * vlm_l2 / sqrt(vlm_numel)

        The trainer unscales mixed-precision gradients before calling this hook.
        ``accelerator`` is accepted for compatibility with the generic policy
        clipping interface.
        """
        _ = accelerator
        if getattr(self.config, "training_stage", "flow") == "next_action":
            return {
                "action_head_clip_applied": 0.0,
                "next_action/gradient_clip_disabled": 1.0,
            }
        if PI05Policy._is_cabo_active(self):
            return {
                "action_head_clip_applied": 0.0,
                "cabo/gradient_clip_disabled": 1.0,
            }
        if not getattr(self.config, "clip_action_head_by_vlm", False):
            return {}

        # Call the class helpers explicitly so the lightweight policy stand-ins used by optimizer
        # tests need only provide ``model`` and ``config``.
        action_parameters = _unique_parameters_with_grad(PI05Policy._action_modules(self))
        vlm_parameters = _unique_parameters_with_grad(PI05Policy._vlm_modules(self))

        action_norm, action_numel = _gradient_l2_norm_and_numel(action_parameters)
        vlm_norm, vlm_numel = _gradient_l2_norm_and_numel(vlm_parameters)

        if action_norm is None or vlm_norm is None or action_numel == 0 or vlm_numel == 0:
            return {
                "action_head_clip_applied": 0.0,
                "action_head_clip_skipped_missing_group": 1.0,
                "action_head_grad_numel": float(action_numel),
                "vlm_grad_numel": float(vlm_numel),
            }

        action_norm_before = float(action_norm.item())
        vlm_norm_value = float(vlm_norm.item())
        action_rms_before = action_norm_before / math.sqrt(action_numel)
        vlm_rms = vlm_norm_value / math.sqrt(vlm_numel)
        ratio = float(getattr(self.config, "action_head_grad_clip_ratio", 1.0))
        max_action_rms = ratio * vlm_rms

        finite = math.isfinite(action_rms_before) and math.isfinite(max_action_rms)
        clip_applied = finite and action_rms_before > max_action_rms
        if clip_applied:
            # Scale only action-side gradients. Using the already-computed RMS ratio avoids a second
            # norm calculation and makes the target explicit; VLM gradients are never written to.
            action_clip_scale = max_action_rms / action_rms_before
            for parameter in action_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(action_clip_scale)

        action_norm_after_tensor, _ = _gradient_l2_norm_and_numel(action_parameters)
        action_norm_after = (
            float(action_norm_after_tensor.item()) if action_norm_after_tensor is not None else float("nan")
        )
        action_rms_after = action_norm_after / math.sqrt(action_numel)
        clip_scale = action_norm_after / action_norm_before if action_norm_before > 0.0 else 1.0

        return {
            "action_head_grad_rms_before_clip": action_rms_before,
            "action_head_grad_rms_after_clip": action_rms_after,
            "vlm_grad_rms": vlm_rms,
            "action_head_grad_norm_before_clip": action_norm_before,
            "action_head_grad_norm_after_clip": action_norm_after,
            "vlm_grad_norm": vlm_norm_value,
            "action_head_clip_threshold_rms": max_action_rms,
            "action_head_clip_scale": clip_scale,
            "action_head_clip_applied": float(clip_applied),
            "action_head_clip_skipped_missing_group": 0.0,
            "action_head_clip_skipped_nonfinite": float(not finite),
            "action_head_grad_numel": float(action_numel),
            "vlm_grad_numel": float(vlm_numel),
        }

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        if self.config.training_stage == "next_action":
            raise RuntimeError(
                "Action inference is unavailable for a frozen-VLM Stage-1 checkpoint; "
                "load it with training_stage='flow' first"
            )
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        if self.config.training_stage == "next_action":
            raise RuntimeError(
                "Action inference is unavailable for a frozen-VLM Stage-1 checkpoint; "
                "load it with training_stage='flow' first"
            )
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def predict_action_chunk_with_context(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> tuple[Tensor, Tensor]:
        """Predict actions and return a pooled contextual PI0.5 prefix representation."""
        if self.config.training_stage == "next_action":
            raise RuntimeError(
                "Action inference is unavailable for a frozen-VLM Stage-1 checkpoint; "
                "load it with training_stage='flow' first"
            )
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions, prefix_features = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            return_prefix_features=True,
            **kwargs,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim], prefix_features

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_loss_weights: Tensor | None = None,
        action_key_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if reduction not in ("mean", "none"):
            raise ValueError(f"Unsupported loss reduction: {reduction!r}")

        batch.pop("_pi05_action_loss_weights", None)
        is_stage1 = self.config.training_stage == PI05TrainingStage.NEXT_ACTION
        if is_stage1 and any(x is not None for x in (noise, time, action_loss_weights, action_key_mask)):
            raise ValueError("Explicit flow inputs are supported only in observation-flow training")
        if is_stage1 and not self._stage1_bridge_active:
            actions = self.prepare_action(batch)
            if "action_is_pad" not in batch:
                raise ValueError("action_is_pad is required for action-only flow pretraining")
            action_is_pad = batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
            self.model._validate_inpainting_inputs(actions, action_is_pad)
            inpainting_mask = self.model.sample_inpainting_mask(action_is_pad)
            # The trainer queries the DDP denominator immediately after forward. Exposing this
            # ephemeral mask on the current batch keeps normalization exact for the sampled mask.
            batch["pi05_inpainting_mask"] = inpainting_mask
            batch[_PI05_LOSS_VALID_STEPS_KEY] = inpainting_mask
            losses = self.model.forward(
                actions=actions,
                action_is_pad=action_is_pad,
                inpainting_mask=inpainting_mask,
                objective=PI05TrainingObjective.ACTION_INPAINTING,
            )

            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
            masked_valid = inpainting_mask.to(device=losses.device)
            loss, per_sample_loss, loss_per_dim = _reduce_action_losses(losses, masked_valid)
            masked_valid_count = masked_valid.sum()
            valid_action_count = (~action_is_pad).sum()
            loss_dict = {
                "loss": loss.detach().item(),
                "flow_inpainting/loss": loss.detach().item(),
                "flow_inpainting/masked_valid_count": float(masked_valid_count.item()),
                "flow_inpainting/masked_valid_fraction": float(
                    masked_valid_count.item() / max(valid_action_count.item(), 1)
                ),
                "stage1/bridge_active": 0.0,
                "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            }
            if reduction == "none":
                return per_sample_loss, loss_dict
            return loss, loss_dict

        return self._forward_observation_flow(
            batch,
            reduction=reduction,
            stage1_bridge=is_stage1,
            noise=noise,
            time=time,
            action_loss_weights=action_loss_weights,
            action_key_mask=action_key_mask,
        )

    def _forward_observation_flow(
        self,
        batch: dict[str, Tensor],
        *,
        reduction: str,
        stage1_bridge: bool,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_loss_weights: Tensor | None = None,
        action_key_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute the formal observation-conditioned flow loss for Stage 2 or its Stage-1 bridge."""
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise=noise,
            time=time,
            objective=PI05TrainingObjective.OBSERVATION_FLOW,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        valid_steps = _valid_action_steps(batch)
        # Always overwrite the runtime denominator mask so a reused batch cannot retain an
        # inpainting mask after the curriculum switches to full flow.
        batch.pop("pi05_inpainting_mask", None)
        batch.pop("_pi05_action_loss_weights", None)
        batch[_PI05_LOSS_VALID_STEPS_KEY] = valid_steps
        # The disabled/B arm uses the original reduction exactly, including its graph.
        loss, per_sample_loss, loss_per_dim = _reduce_action_losses(losses, valid_steps)
        unweighted_loss = loss.detach()
        if action_loss_weights is not None:
            from lerobot.utils.keyframe_loss import weighted_action_loss

            loss, per_sample_loss, loss_per_dim = weighted_action_loss(
                losses, valid_steps, action_loss_weights
            )
            batch["_pi05_action_loss_weights"] = action_loss_weights.to(
                device=losses.device, dtype=torch.float32
            )

        loss_dict = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "valid_action_fraction": valid_steps.float().mean().item(),
            "all_padding_samples": float((~valid_steps.any(dim=1)).sum().item()),
        }
        if action_key_mask is not None:
            from lerobot.utils.keyframe_loss import loss_splits

            loss_dict.update(loss_splits(losses, valid_steps, action_key_mask.to(losses.device)))
            loss_dict["keyframe/unweighted_mse"] = unweighted_loss.item()
            loss_dict["keyframe/weighted_mse"] = loss.detach().item()
        if stage1_bridge:
            loss_dict.update(
                {
                    "stage1/bridge_active": 1.0,
                    "stage1/observation_flow_loss": loss.detach().item(),
                }
            )

        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def get_distributed_loss_normalizer(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the local number of valid scalar loss terms for DDP reweighting."""
        valid_steps = _valid_action_steps(batch)
        if "_pi05_action_loss_weights" in batch:
            weights = batch["_pi05_action_loss_weights"]
            if weights.shape != valid_steps.shape:
                raise ValueError("Invalid runtime action loss weights")
            action_dim = self.config.output_features[ACTION].shape[0]
            return torch.where(valid_steps, weights, 0).sum(dtype=torch.float32) * action_dim
        if self.config.training_stage == PI05TrainingStage.NEXT_ACTION and self._stage1_bridge_active:
            # The bridge always optimizes every valid action step. Prefer the runtime phase over a
            # temporary mask that may have been left on a deliberately reused inpainting batch.
            valid_count = valid_steps.sum(dim=1)
        elif _PI05_LOSS_VALID_STEPS_KEY in batch:
            loss_valid_steps = batch[_PI05_LOSS_VALID_STEPS_KEY]
            if not isinstance(loss_valid_steps, Tensor):
                raise TypeError(
                    f"{_PI05_LOSS_VALID_STEPS_KEY} must be a tensor, got {type(loss_valid_steps).__name__}"
                )
            if loss_valid_steps.shape != valid_steps.shape:
                raise ValueError(
                    f"{_PI05_LOSS_VALID_STEPS_KEY} must have shape {tuple(valid_steps.shape)}, "
                    f"got {tuple(loss_valid_steps.shape)}"
                )
            valid_count = loss_valid_steps.to(dtype=torch.bool).sum(dim=1)
        elif self.config.training_stage == PI05TrainingStage.NEXT_ACTION:
            if "pi05_inpainting_mask" in batch:
                valid_count = batch["pi05_inpainting_mask"].to(dtype=torch.bool).sum(dim=1)
            elif self.config.next_action_full_mask_probability == 1.0:
                valid_count = valid_steps.sum(dim=1)
            else:
                # Before the first forward (or in standalone diagnostics), partial-mask objectives
                # fall back to their fixed-size inpainting denominator.
                valid_count = valid_steps.sum(dim=1).clamp(max=self.config.next_action_masked_steps)
        else:
            valid_count = valid_steps.sum(dim=1)
        action_dim = self.config.output_features[ACTION].shape[0]
        return valid_count.sum(dtype=torch.float32) * action_dim

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Use PEFT to save both prompt banks without adding backbone adapters."""
        return {
            # PEFT's DUMMY_TARGET_MODULES sentinel permits a modules_to_save-only adapter.
            "target_modules": "dummy-target-modules",
            "modules_to_save": ["model.vlm_prompt_tokens", "model.prompt_tokens"],
        }

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        # Even custom adapter settings must save both prompt banks.
        modules_to_save = list(peft_config.modules_to_save or [])
        for name in ("model.vlm_prompt_tokens", "model.prompt_tokens"):
            if name not in modules_to_save:
                modules_to_save.append(name)
        peft_config.modules_to_save = modules_to_save

    def wrap_with_peft(self, peft_config=None, peft_cli_overrides: dict | None = None):
        if getattr(self.config, "train_action_projections", False):
            raise ValueError("train_action_projections requires full checkpoints, not prompt-only PEFT saves")
        policy = super().wrap_with_peft(peft_config, peft_cli_overrides)
        # Freeze custom backbone/projection adapters as well; only active prompt copies train.
        self.model._freeze_vlm()
        return policy
