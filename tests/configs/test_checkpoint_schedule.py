"""Checkpoint selection must survive resume and override periodic saves."""

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig


@pytest.mark.parametrize(
    "save_steps,save_freq,start,expected",
    [
        ([6000, 9000, 12000], 3000, 1, [6000, 9000, 12000]),
        ([6000, 9000], 1, 1, [6000, 9000, 12000]),
        ([6000, 9000, 12000], 3000, 6001, [9000, 12000]),
        ([6000, 9000, 12000], 3000, 9001, [12000]),
        ([], 3000, 1, [12000]),
        (None, 3000, 1, [3000, 6000, 9000, 12000]),
        (None, 5000, 1, [5000, 10000, 12000]),
    ],
)
def test_selected_checkpoints(save_steps, save_freq, start, expected):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/data"), steps=12000, save_steps=save_steps, save_freq=save_freq
    )
    cfg.validate_checkpoint_schedule()
    assert [step for step in range(start, 12001) if cfg.should_save_checkpoint(step)] == expected
    assert not cfg.should_save_checkpoint(0)
    assert not cfg.should_save_checkpoint(12001)
    cfg.save_checkpoint = False
    assert not any(cfg.should_save_checkpoint(step) for step in range(start, 12001))


@pytest.mark.parametrize("save_steps", [[0], [-1], [12001], [6000.0], [True], "6000"])
def test_invalid_save_steps(save_steps):
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/data"), steps=12000, save_steps=save_steps)
    with pytest.raises(ValueError, match="save_steps"):
        cfg.validate_checkpoint_schedule()
