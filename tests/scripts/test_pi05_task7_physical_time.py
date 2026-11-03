"""Experiment II: variable-time labels and temporal encoding regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts import pi05_gripper_event as legacy
from lerobot.scripts import pi05_task7_cadence as cadence
from lerobot.scripts import pi05_task7_physical_time_experiment as experiment
from lerobot.scripts.pi05_task7_cadence_eval import flow_noise
from lerobot.utils.pdet_order import load_plan


def test_index_and_physical_features_respond_differently_to_rate():
    encoder = RelativeActionTimeEncoding(width=16, encoding_dim=32, fps=10)
    x = torch.randn(2,50,16)
    std = torch.tensor([0.1, 0.2])
    index = encoder.features(50, x.device, delta_t=std, basis="index")
    physical = encoder.features(50, x.device, delta_t=std, basis="physical")
    global_dt = encoder.features(50, x.device, delta_t=std, basis="global_dt")
    assert index.shape == (50,32)
    assert physical.shape == global_dt.shape == (2,50,32)
    torch.testing.assert_close(physical[0], index)
    assert not torch.equal(physical[1], index)
    assert not torch.equal(physical[0,4], physical[1,4])
    torch.testing.assert_close(global_dt[:,0], global_dt[:,49])
    assert not torch.equal(global_dt[0,0], global_dt[1,0])
    torch.testing.assert_close(encoder(x, delta_t=std, basis="physical"), x, rtol=0, atol=0)
    with torch.no_grad():
        encoder.weight.copy_(torch.randn_like(encoder.weight))
    assert not torch.equal(
        encoder(x, delta_t=std, basis="physical")[0],
        encoder(x, delta_t=std, basis="physical")[1],
    )


def test_cadence_time_requires_explicit_correct_shape_and_positive_dt():
    enc = RelativeActionTimeEncoding(width=16, encoding_dim=32, fps=10)
    x = torch.randn(2,50,16)
    with pytest.raises(ValueError, match="requires pi05_action_dt"):
        enc(x, basis="physical")
    with pytest.raises(ValueError, match="differs from action batch"):
        enc(x, basis="physical", delta_t=torch.tensor([0.1]))
    with pytest.raises(ValueError, match="finite positive"):
        enc(x, basis="global_dt", delta_t=torch.tensor([-0.1,0.2]))
    with pytest.raises(ValueError, match="Unknown"):
        enc(x, basis="unexpected")


def test_config_guards_and_legacy_default():
    assert PI05Config().action_time_basis == "index"
    with pytest.raises(ValueError, match="action_time_basis"):
        PI05Config(action_time_basis="timestamp_from_future")
    with pytest.raises(ValueError, match="requires input temporal encoding"):
        PI05Config(action_time_basis="physical")
    args = {
        "train_action_expert": True,
        "num_vlm_prompt_tokens": 2,
        "num_prompt_tokens": 0,
        "action_time_encoding": True,
        "cabo_enabled": False,
        "use_peft": False,
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
    }
    for basis in ("index","global_dt","physical"):
        assert PI05Config(**args, action_time_basis=basis).action_time_basis == basis


@pytest.fixture
def demo(tmp_path):
    ds = tmp_path / "libero"
    (ds / "data").mkdir(parents=True)
    (ds / "meta").mkdir()
    count = 220
    actions = np.zeros((count,7), dtype=np.float32)
    actions[:,0] = np.arange(count) / 100.0
    actions[:,-1] = -1
    for j in (20, 80, 150):
        actions[j:j+14,-1] = 1
    pq.write_table(pa.table({
        "index": range(count),
        "episode_index": [0]*count,
        "frame_index": list(range(count)),
        "task_index": [28]*count,
        "action": actions.tolist(),
    }), ds/"data"/"part.parquet")
    pq.write_table(pa.table({
        "task_index":[28],
        "task":[legacy.LONG_TASKS[7]],
    }), ds/"meta"/"tasks.parquet")
    (ds/"meta"/"info.json").write_text(json.dumps({"fps":10, "total_frames":count}))
    base = tmp_path/"source_uniform"
    legacy.prepare(
        ds, base, task_id=7, episode_index=0,
        steps=1000, batch_size=32, world_size=2,
    )
    return ds, base, actions


@pytest.fixture
def prepared(demo, tmp_path):
    ds, base, actions = demo
    path = tmp_path/"cadence"
    record = cadence.prepare(path, source_dir=base)
    return path, record, actions


def test_cadence_plan_matches_1000_steps_and_is_deterministic(prepared):
    path, record, actions = prepared
    again = cadence.check(path)
    assert again == record
    plan = load_plan(path/"uniform.npz")
    with np.load(path/"cadence.npz", allow_pickle=False) as d:
        strides = d["strides"]
    assert plan["manifest"]["steps"] == 1000
    assert len(plan["index"]) == len(strides) == 64000
    assert int((strides == 1).sum()) == int((strides == 2).sum()) == 32000
    assert np.all(plan["index"] + 49*strides <= record["episode_last"])
    assert set(plan["index"]).isdisjoint(record["heldout_starts"])
    assert record["stride3_heldout_starts"]
    assert len(record["stride3_heldout_starts"]) < len(record["heldout_starts"])
    assert record["fps"] == 10.0


def test_cadence_labels_are_true_future_actions_not_adjacent_frames(prepared):
    path, record, actions = prepared
    start = record["heldout_starts"][0]
    a = cadence.make_action_targets(actions, [start], [1], first=0, last=219)[0]
    b = cadence.make_action_targets(actions, [start], [2], first=0, last=219)[0]
    assert a.shape == b.shape == (50,7)
    np.testing.assert_array_equal(a[:,0], actions[start + np.arange(50),0])
    np.testing.assert_array_equal(b[:,0], actions[start + np.arange(50)*2,0])
    assert b[1,0] != a[1,0]
    with pytest.raises(ValueError, match="crosses"):
        cadence.make_action_targets(actions, [180],[2],first=0,last=219)


class SmallSource:
    def __init__(self, root, actions):
        self.root = root
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return {
            "index": int(index),
            "action": torch.zeros(50,7),
            "action_is_pad": torch.ones(50, dtype=torch.bool),
            "observation.state": torch.ones(8),
        }


def test_planned_dataset_overwrites_full_targets_by_occurrence(prepared, demo):
    path, record, actions = prepared
    ds = SmallSource(demo[0], actions)
    wrapped = cadence.CadencePlannedDataset(ds, path)
    plan = load_plan(path/"uniform.npz")
    with np.load(path/"cadence.npz",allow_pickle=False) as d:
        all_strides = d["strides"]
    for occurrence in (0,1,177):
        index = int(plan["index"][occurrence])
        item = wrapped[(index,occurrence)]
        stride = int(all_strides[occurrence])
        assert item["_pdet_occurrence_id"] == occurrence
        assert item["_cadence_delta_t"] == stride/10
        assert not item["action_is_pad"].any()
        np.testing.assert_array_equal(
            item["action"].numpy(),
            actions[index+stride*np.arange(50)],
        )
    with pytest.raises(ValueError, match="Occurrence"):
        wrapped[(int(plan["index"][0])+1,0)]


def test_four_models_share_one_plan_with_distinct_time_conditioning(prepared, tmp_path):
    path, record, actions = prepared
    record["prepared_root"] = str(path)
    jobs = experiment.enrich_jobs(record, tmp_path/"models")
    assert [j["case"] for j in jobs] == list(experiment.CASES)
    assert all(j["plan"] == path/"uniform.npz" for j in jobs)
    assert all(j["checkpoint"].parts[-2:] == ("001000","pretrained_model") for j in jobs)
    args = SimpleNamespace(
        source=tmp_path/"source",tokenizer=tmp_path/"tokenizer",
        output_root=tmp_path/"models",log_root=tmp_path/"logs",
        dataset_repo_id="libero",gpus="0,1",port=29589,
        dry_run=True,num_workers=0,wandb_mode="offline",
    )
    for job in jobs:
        command, env = experiment.training_command(job, record, args, tmp_path/"prompt")
        assert f"--policy.action_time_basis={job['time_basis']}" in command
        assert f"--policy.action_time_encoding={str(job['time_encoding']).lower()}" in command
        assert env["PI05_CADENCE_PLAN"] == str(path)
        assert env["FLOW_STEPS"] == "1000"
        assert env["GPU_IDS"] == "0,1"


def test_eval_flow_noise_is_model_independent():
    a = flow_noise(3000, 15, 2, (1,50,32))
    torch.testing.assert_close(a, flow_noise(3000,15,2,(1,50,32)), rtol=0, atol=0)
    assert not torch.equal(a, flow_noise(3000,15,1,(1,50,32)))
    assert not torch.equal(a, flow_noise(3001,15,2,(1,50,32)))


def test_training_loop_pops_and_restores_delta_after_processor():
    root = Path(__file__).resolve().parents[2]
    source = (root/"src/lerobot/scripts/lerobot_train.py").read_text()
    assert 'batch.pop("_cadence_delta_t", None)' in source
    assert 'batch["pi05_action_dt"] = torch.as_tensor(' in source
    assert 'CadencePlannedDataset(dataset, prepared)' in source
    policy = (root/"src/lerobot/policies/pi05/modeling_pi05.py").read_text()
    assert 'action_delta_t = batch.get("pi05_action_dt")' in policy
    assert "action_delta_t=action_delta_t" in policy


def test_eval_runner_is_offline_and_two_phases():
    root = Path(__file__).resolve().parents[2]
    sh = (root/"run_eval_libero_task7_physical_time_2gpu.sh").read_text()
    assert sh.index("phase rope index") < sh.index("phase global_dt physical")
    assert "summarize all" in sh
    evaluator = (root/"src/lerobot/scripts/pi05_task7_cadence_eval.py").read_text()
    assert "policy.forward(batch, noise=noise, time=time)" in evaluator
    assert "No action execution" in evaluator
