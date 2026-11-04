"""CPU tests for task7 temporal encoder evidence and paired rollout diagrams.

No checkpoint weights, MuJoCo simulator, online service, or GPU is used.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.scripts import pi05_task7_time_evidence as evidence


def test_paired_outcomes_with_crossed_successes():
    rope=[True,True,False,False,False,True]
    timed=[True,False,True,False,True,True]
    observed=evidence.paired_counts(rope,timed)
    assert observed=={
        "both_success":2, "both_failure":1,
        "time_only_success":2,"rope_only_success":1,
        "net_time_successes":1,
    }
    with pytest.raises(ValueError,match="equal"):
        evidence.paired_counts(rope,timed[:-1])


def test_extracts_only_raw_task7_bool_episodes(tmp_path):
    output=tmp_path/"eval_info.json"
    content={"per_task":[{"task_group":"libero_10","task_id":7,
                         "metrics":{"successes":[True,False,True,False]}}]}
    output.write_text(json.dumps(content))
    assert evidence._task7_successes(output,expected=4).tolist()==[1,0,1,0]
    content["per_task"][0]["task_id"]=8
    output.write_text(json.dumps(content))
    with pytest.raises(ValueError,match="LIBERO-long task7"):
        evidence._task7_successes(output,expected=4)
    content["per_task"][0]["task_id"]=7
    content["per_task"][0]["metrics"]["successes"]=[True,1,False,False]
    output.write_text(json.dumps(content))
    with pytest.raises(ValueError,match="boolean"):
        evidence._task7_successes(output,expected=4)


def test_stable_flow_noise_identical_within_and_different_across_anchors():
    same=evidence.stable_noise(3000,123,(1,50,32))
    torch.testing.assert_close(same,evidence.stable_noise(3000,123,(1,50,32)),atol=0,rtol=0)
    assert not torch.equal(same,evidence.stable_noise(3000,124,(1,50,32)))
    assert not torch.equal(same,evidence.stable_noise(4000,123,(1,50,32)))
    assert same.std()>0.5


def test_probe_selects_exclusive_other_task7_demos():
    index=np.arange(420)
    metadata={
        "episode_index":np.repeat([7,8,9],140),
        "frame_index":np.tile(np.arange(140),3),
        "task_index":np.repeat([11,11,11],140),
    }
    result=evidence.select_nontraining_anchors(
        metadata,task_index=11,train_episode=7,count=4,
    )
    assert len(result)==4
    assert all(ep in (8,9) for ep,_ in result)
    assert all(140<=idx<=420-50 for _,idx in result)
    assert len({idx for _,idx in result})==4
    with pytest.raises(ValueError,match="No other complete"):
        evidence.select_nontraining_anchors(
            {key:value[:140] for key,value in metadata.items()},
            task_index=11,train_episode=7,count=2,
        )


def test_probe_does_not_cross_other_episode_frame_boundary():
    metadata={
        "episode_index":np.repeat([7,8],75),
        "frame_index":np.r_[np.arange(75),np.arange(75)],
        "task_index":np.repeat([11,11],75),
    }
    anchors=evidence.select_nontraining_anchors(
        metadata,task_index=11,train_episode=7,count=8,
    )
    assert len(anchors)>=1
    assert all(idx+49<150 for _,idx in anchors)


def test_temporal_interventions_are_reversible_and_do_not_touch_weights():
    torch.manual_seed(7)
    enc=RelativeActionTimeEncoding(16,32,10.0)
    with torch.no_grad():
        enc.weight.copy_(torch.randn_like(enc.weight)*0.03)
    before=enc.weight.detach().clone()
    x=torch.randn(2,50,16)
    correct=enc(x)
    assert not torch.equal(correct,x)
    with evidence.time_counterfactual(enc,"zero"):
        torch.testing.assert_close(enc(x),x,atol=0,rtol=0)
    torch.testing.assert_close(enc(x),correct,atol=0,rtol=0)
    with evidence.time_counterfactual(enc,"reversed"):
        reversed_features=enc(x)
    assert not torch.equal(reversed_features,correct)
    torch.testing.assert_close(enc(x),correct,atol=0,rtol=0)
    torch.testing.assert_close(enc.weight,before,atol=0,rtol=0)
    with pytest.raises(ValueError,match="Unknown"):
        with evidence.time_counterfactual(enc,"nonexistent"):
            pass


def test_zero_time_residual_is_not_confused_with_rope_position_scramble():
    enc=RelativeActionTimeEncoding(16,32,10.0)
    with torch.no_grad():
        enc.weight.copy_(torch.randn_like(enc.weight)*0.01)
    x=torch.randn(1,50,16)
    with evidence.time_counterfactual(enc,"zero"):
        assert torch.equal(enc(x),x)
    assert enc(x).shape==x.shape
    assert not torch.equal(enc(x),x)


def test_figures_are_data_driven_and_emit_png_svg(tmp_path):
    pytest.importorskip("matplotlib")
    np.random.seed(33)
    matrix=np.random.randint(0,2,size=(4,20),dtype=np.int64)
    files=evidence.plot_success(matrix,[1000,3000],tmp_path)
    assert len(files)==2
    assert all(Path(p).is_file() and Path(p).stat().st_size>500 for p in files)
    residual=np.random.normal(0,0.02,size=(3,50,16))
    files2=evidence.plot_residual(residual,10.0,tmp_path)
    assert len(files2)==2
    mse_base=np.random.uniform(.2,.5,size=(3,50))
    mse_time={
        "correct":mse_base+.01,
        "zero":mse_base+.03,
        "reversed":mse_base+.05,
    }
    pr=np.random.normal(0,1,size=(3,50,7))
    pretend={
        "rope":{"mse":{"correct":mse_base}},
        "time_input":{
            "mse":mse_time,
            "pred":{"correct":pr,"zero":pr+.1,"reversed":pr+.2},
            "ratio":np.abs(np.random.normal(.05,.01,size=(3,50))),
            "residual":residual,
        },
    }
    files3=evidence.plot_probe(pretend,tmp_path)
    assert len(files3)==2
    assert all(Path(p).is_file() and Path(p).stat().st_size>500 for p in files3)


def test_default_data_inference_does_not_auto_retrain_or_execute_robot():
    root=Path(__file__).resolve().parents[2]
    source=(root/"src/lerobot/scripts/pi05_task7_time_evidence.py").read_text()
    assert "time_counterfactual" in source
    assert "model.predict_velocity" in source
    assert "select_nontraining_anchors" in source
    assert 'matplotlib.use("Agg")' in source
    assert '"per_step_mean_mse"' in source
    assert 'if args.mode in ("probe","all")' in source
    assert "action_time_encoder" in source
