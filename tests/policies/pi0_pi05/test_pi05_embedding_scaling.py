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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    PaliGemmaWithExpertModel,
    PI05Pytorch,
)


def test_pi05_image_embedding_preserves_projector_scale():
    projected_features = torch.ones(1, 2, 4, dtype=torch.float32)

    class _PaliGemmaModel:
        def get_image_features(self, image):
            assert image.dtype == torch.float32
            return SimpleNamespace(pooler_output=projected_features)

    fake_model = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=_PaliGemmaModel(),
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)),
        )
    )
    image = torch.ones(1, 3, 2, 2, dtype=torch.bfloat16)

    embedded = PaliGemmaWithExpertModel.embed_image(fake_model, image)

    torch.testing.assert_close(embedded, projected_features.to(torch.bfloat16))
    assert embedded.dtype == image.dtype


def test_pi05_language_embedding_uses_transformers_accessor_without_rescaling():
    tokens = torch.tensor([[1, 2]])
    scaled_embeddings = torch.full((1, 2, 4), 3.0)
    accessor_calls = 0

    def get_input_embeddings():
        nonlocal accessor_calls
        accessor_calls += 1
        return lambda input_tokens: scaled_embeddings + input_tokens.unsqueeze(-1) * 0

    fake_model = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(get_input_embeddings=get_input_embeddings),
            )
        )
    )

    embedded = PaliGemmaWithExpertModel.embed_language_tokens(fake_model, tokens)

    torch.testing.assert_close(embedded, scaled_embeddings)
    assert accessor_calls == 1


def test_pi05_prefix_preserves_image_and_already_scaled_language_embeddings():
    image_embeddings = torch.full((1, 2, 4), 2.0)
    language_embeddings = torch.full((1, 3, 4), 3.0)
    fake_core = SimpleNamespace(
        config=SimpleNamespace(num_vlm_prompt_tokens=0),
        paligemma_with_expert=SimpleNamespace(
            embed_image=lambda _image: image_embeddings,
            embed_language_tokens=lambda _tokens: language_embeddings,
        ),
        _apply_checkpoint=lambda function, *args: function(*args),
    )

    prefix, pad_mask, attention_mask = PI05Pytorch.embed_prefix(
        fake_core,
        images=[torch.zeros(1, 3, 2, 2)],
        img_masks=[torch.tensor([True])],
        tokens=torch.tensor([[1, 2, 3]]),
        masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix[:, :2], image_embeddings)
    torch.testing.assert_close(prefix[:, 2:], language_embeddings)
    torch.testing.assert_close(pad_mask, torch.tensor([[True, True, True, True, False]]))
    assert not attention_mask.any()
