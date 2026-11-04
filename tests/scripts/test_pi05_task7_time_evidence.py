"""CPU tests for task7 temporal encoder evidence and paired rollout diagrams.

No checkpoint weights, MuJoCo simulator, online service, or GPU is used.
"""

import json
from pathlib import Path
from types import SimpleNamespace

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


def _metadata(episodes=9, length=200):
    return {
        "task_index":np.full(episodes*length,11),
        "episode_index":np.repeat(np.arange(episodes),length),
        "frame_index":np.tile(np.arange(length),episodes),
    }


def test_default_selection_covers_early_middle_late_in_each_demo():
    meta=_metadata()
    anchors=evidence.select_nontraining_anchors(meta,task_index=11,train_episode=0)
    descriptions=evidence.describe_anchors(meta,anchors)
    assert len(anchors)==24
    assert {row["episode_index"] for row in descriptions}==set(range(1,9))
    for episode in range(1,9):
        rows=[row for row in descriptions if row["episode_index"]==episode]
        assert [row["frame_index"] for row in rows]==[0,75,150]
        assert [row["phase"] for row in rows]==["early","middle","late"]
        assert all(row["end_frame_inclusive"]<200 for row in rows)
    capped=evidence.select_nontraining_anchors(meta,task_index=11,train_episode=0,count=8)
    # The legacy cap samples the complete coverage, not eight episode starts.
    assert {row["phase"] for row in evidence.describe_anchors(meta,capped)}=={"early","middle","late"}


def test_short_demos_report_overlap_without_crossing_boundaries():
    meta=_metadata(episodes=3,length=75)
    anchors=evidence.select_nontraining_anchors(meta,task_index=11,train_episode=0)
    rows=evidence.describe_anchors(meta,anchors)
    assert len(rows)==6
    assert any(row["overlaps_previous_selected_window"] for row in rows)
    assert all(row["end_frame_inclusive"]<75 for row in rows)
    with pytest.raises(ValueError,match="crosses an episode"):
        evidence.describe_anchors(meta,[(1,140)])


def test_repeat_selects_both_requested_checkpoints(tmp_path):
    args=SimpleNamespace(directory=tmp_path/"plan",output_root=tmp_path/"models",repeat=3,
                         rope_checkpoint=None,time_input_checkpoint=None)
    paths=evidence.resolve_probe_checkpoints(args,{"selection":{"episode_index":7}})
    assert "r3-rope-task7-ep7" in str(paths["rope"])
    assert "r3-time_input-task7-ep7" in str(paths["time_input"])


def test_probe_grid_defaults_legacy_overrides_and_invalid_values():
    assert evidence.probe_grid(SimpleNamespace())==([0.2,0.5,0.8],[3000,4000,5000])
    assert evidence.probe_grid(SimpleNamespace(flow_time=0.5,noise_seed=42))==([0.5],[42])
    assert evidence.probe_grid(SimpleNamespace(flow_times="0.3,0.7",noise_seeds="7,9"))==([0.3,0.7],[7,9])
    for invalid in ("0,0.5","0.3,nan","0.5,0.5","1", ""):
        with pytest.raises(ValueError,match="Flow times|flow times"):
            evidence.parse_flow_times(invalid)
    with pytest.raises(ValueError,match="either --flow"):
        evidence.probe_grid(SimpleNamespace(flow_time=0.5,flow_times="0.2,0.8"))


def test_episode_aggregation_does_not_count_noises_or_windows_as_demos():
    conditions=[
        {"episode_index":1,"global_frame_index":10},
        {"episode_index":1,"global_frame_index":10},
        {"episode_index":1,"global_frame_index":10},
        {"episode_index":1,"global_frame_index":30},
        {"episode_index":2,"global_frame_index":60},
    ]
    episodes,means=evidence.episode_means(np.array([1.,1.,1.,3.,20.]),conditions)
    assert episodes==[1,2]
    np.testing.assert_allclose(means,[2.,20.])
    assert means.mean()==11.0
    with pytest.raises(ValueError,match="nonfinite"):
        evidence.episode_means(np.array([1.,1.,1.,3.,np.nan]),conditions)


def test_alignment_summary_keeps_adverse_results():
    baseline=np.repeat(np.array([[1.],[3.]]),50,axis=1)
    data={"episode_ids":[10,20],"rope":{"mse":{"correct":baseline+.1}},
          "time_input":{"mse":{"correct":baseline+.4,"zero":baseline,"reversed":baseline-.2}}}
    report=evidence.alignment_summary(data)
    assert report["observed_direction"]=="mixed_or_adverse_alignment_evidence"
    assert report["mean_paired_deltas"]["zero_minus_correct"]==pytest.approx(-.4)
    assert report["mean_paired_deltas"]["reversed_minus_correct"]==pytest.approx(-.6)
    assert report["mean_paired_deltas"]["correct_minus_rope"]==pytest.approx(.3)
    assert report["delta_direction_counts"]["zero_minus_correct"]["negative"]==2


def test_comparison_and_checkpoint_training_plan_must_match(tmp_path):
    record={"selection":{"episode_index":7},"dataset_root":"/example/data"}
    plan=tmp_path/"plan"
    plan.mkdir()
    (plan/"manifest.json").write_text(json.dumps(record))
    digest=evidence.previous.file_hash(plan/"manifest.json")
    evidence.validate_comparison_plan({"settings":{"preparation_sha256":digest}},plan)
    evidence.validate_comparison_plan({"lock":{"settings":{"preparation_sha256":digest}}},plan)
    with pytest.raises(ValueError,match="training plan differs"):
        evidence.validate_comparison_plan({"settings":{"preparation_sha256":"different"}},plan)
    checkpoint=tmp_path/"run"/"checkpoints"/"001000"/"pretrained_model"
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError,match="missing training provenance"):
        evidence.validate_training_provenance({"rope":checkpoint},record)
    descriptor=checkpoint.parents[2]/"gripper_event_experiment.json"
    descriptor.write_text(json.dumps({"preparation":record}))
    assert evidence.validate_training_provenance({"rope":checkpoint},record)["rope"]["sha256"]
    descriptor.write_text(json.dumps({"preparation":{"selection":{"episode_index":8}}}))
    with pytest.raises(ValueError,match="training plan differs"):
        evidence.validate_training_provenance({"rope":checkpoint},record)


class _TinyPolicy:
    """A real Torch toy used to check the harness, not a PI0.5/GPU test."""

    def __init__(self, timed=True):
        self.config=SimpleNamespace(action_time_basis="index",output_features={"action":SimpleNamespace(shape=(7,))})
        self.model=SimpleNamespace()
        generator=torch.Generator().manual_seed(99)
        self.model.action_in_proj=torch.nn.Linear(32,16)
        with torch.no_grad():
            self.model.action_in_proj.weight.copy_(torch.randn(16,32,generator=generator)*.03)
            self.model.action_in_proj.bias.zero_()
        encoder=RelativeActionTimeEncoding(16,32,10.0) if timed else None
        if encoder is not None:
            with torch.no_grad():
                encoder.weight.copy_(torch.randn(16,32,generator=generator)*.01)
        self.model.action_time_encoder=encoder
        self.calls=[]

        def predict(images,masks,tokens,token_masks,x_t,t):
            self.calls.append((x_t.detach().clone(),t.detach().clone()))
            projected=self.model.action_in_proj(x_t)
            residual=(encoder(projected)-projected) if encoder is not None else torch.zeros_like(projected)
            return .1*x_t + residual.mean(dim=-1,keepdim=True)

        self.model.predict_velocity=predict

    def prepare_action(self,batch):
        return batch["action"]

    def _preprocess_images(self,batch):
        return [],[]


def _tiny_batch(frame_index=0):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    return {
        "action":torch.linspace(-.4,.6,50*32).reshape(1,50,32)+frame_index/10000,
        OBS_LANGUAGE_TOKENS:torch.zeros(1,3,dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK:torch.ones(1,3,dtype=torch.bool),
    }


def test_one_step_action_estimate_uses_correct_flow_identity():
    policy=_TinyPolicy(timed=False)
    batch=_tiny_batch()
    noise=torch.randn(1,50,32,generator=torch.Generator().manual_seed(6))
    expected_velocity=noise-batch["action"]
    policy.model.predict_velocity=lambda *args:expected_velocity
    result=evidence._probe_predict(policy,batch,noise,.3)
    np.testing.assert_allclose(result["clean_estimate"],batch["action"][0,:,:7].numpy(),atol=1e-7)
    np.testing.assert_allclose(result["per_step_mse"],0,atol=0)
    policy.config.action_time_basis="physical"
    with pytest.raises(ValueError,match="basis=index"):
        evidence._probe_predict(policy,batch,noise,.3)


def test_multigrid_forward_pairs_inputs_across_all_interventions(monkeypatch,tmp_path):
    policy=_TinyPolicy()
    before=policy.model.action_time_encoder.weight.detach().clone()
    monkeypatch.setattr(evidence,"_load_policy_and_processor",lambda *args:(policy,None))
    monkeypatch.setattr(evidence,"_batch_for_index",lambda dataset,actions,idx,processor:_tiny_batch(idx))
    args=SimpleNamespace(tokenizer=tmp_path,flow_times="0.2,0.8",noise_seeds="11,12")
    result=evidence._compute_probe_for_model(
        "time_input",tmp_path,None,None,[(1,100),(2,200)],args,torch.device("cpu"),
        interventions=("correct","zero","reversed"),
    )
    assert len(result["conditions"])==8
    assert result["mse"]["correct"].shape==(8,50)
    assert result["clean_estimate"]["reversed"].shape==(8,50,7)
    assert len(policy.calls)==24
    for start in range(0,len(policy.calls),3):
        for later in (1,2):
            torch.testing.assert_close(policy.calls[start][0],policy.calls[start+later][0],atol=0,rtol=0)
            torch.testing.assert_close(policy.calls[start][1],policy.calls[start+later][1],atol=0,rtol=0)
    torch.testing.assert_close(policy.model.action_time_encoder.weight,before,atol=0,rtol=0)
    assert not np.array_equal(result["pred"]["correct"],result["pred"]["zero"])


def test_complete_cpu_probe_exports_episode_summary_and_honest_figure(monkeypatch,tmp_path):
    pytest.importorskip("matplotlib")
    record={"dataset_root":str(tmp_path/"dataset"),"fps":10.,"selection":{"episode_index":0,"task_index":11}}
    checkpoints={}
    for case in ("rope","time_input"):
        path=tmp_path/case/"checkpoints"/"001000"/"pretrained_model"
        path.mkdir(parents=True)
        (path/"model.safetensors").write_bytes(b"test fixture: tiny in-memory model")
        config={"train_action_expert":True,"num_vlm_prompt_tokens":2,"num_prompt_tokens":0,
                "chunk_size":50,"action_time_encoding":case=="time_input","action_time_injection":"input",
                "action_time_basis":"index","normalization_mapping":{},"input_features":{},"output_features":{}}
        (path/"config.json").write_text(json.dumps(config))
        (path.parents[2]/"gripper_event_experiment.json").write_text(json.dumps({"preparation":record}))
        checkpoints[case]=path
    args=SimpleNamespace(
        flow_times="0.2,0.5",noise_seeds="11,12",allow_cpu=True,gpu=0,repeat=2,
        directory=tmp_path/"plan",output_root=tmp_path,output=tmp_path/"analysis",tokenizer=tmp_path,
        rope_checkpoint=checkpoints["rope"],time_input_checkpoint=checkpoints["time_input"],
        samples=None,probe_demos=2,windows_per_demo=3,
    )
    meta=_metadata(episodes=3)
    anchors=evidence.select_nontraining_anchors(meta,task_index=11,train_episode=0,demos=2)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(evidence,"_probe_data",lambda args,record:(anchors,None,None,meta))
    monkeypatch.setattr(evidence,"_batch_for_index",lambda dataset,actions,idx,processor:_tiny_batch(idx))
    monkeypatch.setattr(evidence,"_load_policy_and_processor",
                        lambda path,*args:(_TinyPolicy(timed="time_input" in Path(path).parts),None))
    result=evidence.probe(args,record)
    summary=result["time_alignment_summary"]
    assert (summary["n_episodes"],summary["n_anchors"],summary["n_noise"],summary["n_flow"],summary["n_conditions"])==(2,6,2,2,24)
    assert summary["aggregation_unit"]=="demonstration"
    assert len(summary["paired_episode_deltas"]["zero_minus_correct"])==2
    assert len(result["paired_condition_mse"]["correct"])==24
    assert len(result["paired_anchor_mse"]["correct"])==6
    assert summary["representative_condition"]["episode_index"]==1
    assert summary["representative_condition"]["frame_index"]==75
    assert summary["representative_condition"]["flow_time"]==.5
    assert summary["representative_condition"]["noise_seed"]==11
    assert len(result["figure_paths"])==6
    assert all(Path(path).stat().st_size>500 for path in result["figure_paths"])
    saved=json.loads((args.output/"time_alignment_evidence.json").read_text())
    assert saved["mean_paired_deltas"]==summary["mean_paired_deltas"]
    arrays=np.load(args.output/"offline_probe_arrays.npz")
    assert arrays["correct_velocity"].shape==(24,50,7)
    assert arrays["episode_mean_time_residual"].shape==(2,50,16)

