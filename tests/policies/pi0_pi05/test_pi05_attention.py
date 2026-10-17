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

"""Compare PI0.5 SDPA with Gemma eager attention, including grouped KV heads."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

pytest.importorskip("transformers")

from lerobot.policies.pi05.attention_pi05 import pi05_attention_forward  # noqa: E402
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE  # noqa: E402


def _additive_mask(kind, query_offset):
    if kind == "none":
        return None
    # Blocks O(2), V(2), E(1), A(3); these are bidirectional, not causal.
    blocks = torch.tensor([0, 0, 1, 1, 2, 3, 3, 3])
    visibility = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [True, True, True, True],
        ]
    )
    allowed = visibility[blocks[:, None], blocks[None, :]].expand(2, -1, -1).clone()
    if kind == "padding":
        allowed[1, :, -1] = False
        allowed[1, -1, :] = False
        allowed[1, :2, :] = False
        allowed[1, :, :2] = False
    return torch.where(allowed[:, None, query_offset:, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)


def _compare_attention(dtype, kv_heads, query_offset, mask, *, valid_queries_only):
    torch.manual_seed(73)
    module = SimpleNamespace(num_key_value_groups=8 // kv_heads, training=True)
    inputs = (
        torch.randn(2, 8, 8 - query_offset, 16, dtype=dtype, requires_grad=True),
        torch.randn(2, kv_heads, 8, 16, dtype=dtype, requires_grad=True),
        torch.randn(2, kv_heads, 8, 16, dtype=dtype, requires_grad=True),
    )
    eager, weights = pi05_attention_forward(module, *inputs, mask, 0.25, implementation="eager")
    actual, actual_weights = pi05_attention_forward(module, *inputs, mask, 0.25)
    assert actual_weights is None
    assert weights.shape == (2, 8, 8 - query_offset, 8)
    assert actual.shape == (2, 8 - query_offset, 8, 16)
    assert actual.dtype == dtype
    tolerance = {"rtol": 0.02, "atol": 0.015625} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(actual, eager, **tolerance)

    upstream = torch.randn_like(eager)
    if valid_queries_only and mask is not None:
        valid_queries = mask.eq(0).any(dim=-1).squeeze(1)
        upstream = upstream.masked_fill(~valid_queries[:, :, None, None], 0)
    eager_grads = torch.autograd.grad((eager * upstream).mean(), inputs)
    actual_grads = torch.autograd.grad((actual * upstream).mean(), inputs)
    tolerance = {"rtol": 0.03, "atol": 3e-5} if dtype == torch.bfloat16 else {"rtol": 2e-5, "atol": 1e-8}
    for actual_grad, eager_grad in zip(actual_grads, eager_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, eager_grad, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 2, 8])
@pytest.mark.parametrize("query_offset", [0, 4])
@pytest.mark.parametrize("mask_kind", ["none", "blocks", "padding"])
def test_sdpa_matches_eager_for_valid_query_outputs_and_gradients(dtype, kv_heads, query_offset, mask_kind):
    # Fully masked rows are padding: compare their output too, but do not optimize it.
    _compare_attention(
        dtype, kv_heads, query_offset, _additive_mask(mask_kind, query_offset), valid_queries_only=True
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_math_sdpa_matches_eager_gradients_even_for_fully_masked_queries(dtype):
    mask = torch.full((2, 1, 8, 8), OPENPI_ATTENTION_MASK_VALUE, dtype=torch.float32)
    # CPU Flash's finite-sentinel backward need not match eager for nonzero gradients
    # on a fully masked row. Math is an independent reference for that unused case.
    with sdpa_kernel(SDPBackend.MATH):
        _compare_attention(dtype, 1, 0, mask, valid_queries_only=False)


def test_invalid_attention_implementation_is_rejected():
    tensor = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="Unsupported PI0.5 attention implementation"):
        pi05_attention_forward(SimpleNamespace(), tensor, tensor, tensor, None, 1.0, implementation="unknown")
