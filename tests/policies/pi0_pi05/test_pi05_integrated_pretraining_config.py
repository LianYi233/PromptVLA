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

import draccus
import pytest

pytest.importorskip("transformers")

from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402


def test_pi05_integrated_next_action_pretraining_defaults_to_750_plus_250_steps():
    config = PI05Config()

    assert config.next_action_pretrain_steps == 1_000
    assert config.next_action_bridge_steps == 250
    assert config.next_action_masked_steps == 40
    assert config.next_action_full_mask_probability == pytest.approx(0.0)
    assert config.next_action_pretraining_active


@pytest.mark.parametrize(
    ("kwargs", "expected_steps"),
    [
        ({"next_action_pretrain_steps": 0}, 0),
        ({"training_stage": "next_action"}, 1_000),
    ],
)
def test_pi05_integrated_next_action_pretraining_can_be_disabled_and_never_recurses(kwargs, expected_steps):
    config = PI05Config(**kwargs)

    assert config.next_action_pretrain_steps == expected_steps
    assert not config.next_action_pretraining_active


def test_pi05_rejects_negative_integrated_next_action_pretraining_steps():
    with pytest.raises(ValueError, match="next_action_pretrain_steps must be non-negative"):
        PI05Config(next_action_pretrain_steps=-1)


def test_pi05_bridge_schema_validation_remains_legacy_loadable():
    with pytest.raises(ValueError, match="next_action_bridge_steps must be non-negative"):
        PI05Config(next_action_bridge_steps=-1)

    # Cross-field validation happens when integrated Stage 1 is built. Keeping schema decoding
    # permissive lets older checkpoint configs that predate the bridge field remain loadable.
    legacy_short_config = PI05Config(next_action_pretrain_steps=249, next_action_bridge_steps=250)
    assert legacy_short_config.next_action_pretrain_steps == 249

    disabled = PI05Config(next_action_pretrain_steps=0, next_action_bridge_steps=250)
    assert not disabled.next_action_pretraining_active


def test_pi05_integrated_next_action_pretraining_steps_decode_from_cli():
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=pi05",
            "--policy.next_action_pretrain_steps=1200",
            "--policy.next_action_bridge_steps=300",
        ],
    )

    assert isinstance(config.policy, PI05Config)
    assert config.policy.next_action_pretrain_steps == 1_200
    assert config.policy.next_action_bridge_steps == 300
    assert config.policy.next_action_pretraining_active
