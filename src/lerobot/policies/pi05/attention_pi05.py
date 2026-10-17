#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Attention backends for PI0.5's explicit, noncausal visibility masks."""

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def pi05_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    *,
    implementation: Literal["sdpa", "eager"] = "sdpa",
) -> tuple[Tensor, Tensor | None]:
    """Return attention in [batch, query tokens, heads, head dimension] order.

    The additive mask remains float32 even for bfloat16 queries. In particular,
    preserve PI0.5's finite negative padding value instead of converting it to
    an infinite value with different fully masked-row semantics. Fully masked
    rows represent padding and their outputs must not contribute to the loss.
    """
    if implementation == "eager":
        # Transformers is optional; the optimized backend only depends on PyTorch.
        from transformers.models.gemma.modeling_gemma import eager_attention_forward

        return eager_attention_forward(module, query, key, value, attention_mask, scaling)
    if implementation != "sdpa":
        raise ValueError(f"Unsupported PI0.5 attention implementation: {implementation!r}")

    enable_gqa = query.shape[1] != key.shape[1]
    if enable_gqa and key.shape[1] == 1 and (query.device.type != "cuda" or query.dtype == torch.float32):
        # A stride-zero view avoids copying PI0.5's single KV head. Matching head
        # counts also permits fused backends that lack native GQA (notably FP32).
        key = key.expand(-1, query.shape[1], -1, -1)
        value = value.expand(-1, query.shape[1], -1, -1)
        enable_gqa = False

    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=enable_gqa,
    )
    return output.transpose(1, 2).contiguous(), None
