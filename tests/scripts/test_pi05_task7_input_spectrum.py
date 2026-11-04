"""Small deterministic checks for the matched 1000-step time-input spectral study.

Does not load PaliGemma weights, build a simulator or allocate a GPU.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts import pi05_task7_input_spectrum as spectra
from lerobot.scripts import pi05_task7_timepath_experiment as original
from lerobot.scripts import pi05_gripper_event_experiment as previous


def config_params(**kw):
    return dict(
        train_action_expert=True, num_vlm_prompt_tokens=2,
        num_prompt_tokens=0, action_time_encoding=True,
        cabo_enabled=False, use_peft=False,
        next_action_pretrain_steps=0, next_action_bridge_steps=0,
        **kw,
    )


def test_default_input_features_bitwise_match_original_point_one_to_ten_hz():
    enc=RelativeActionTimeEncoding(1024,32,10)
    freq=torch.logspace(-1,1,16)
    seconds=torch.arange(50,dtype=torch.float32)/10
    phases=2*torch.pi*seconds[:,None]*freq[None,:]
    expected=torch.cat([phases.sin(),phases.cos()],dim=-1)
    torch.testing.assert_close(enc.features(50, torch.device("cpu")),expected,rtol=0,atol=0)
    assert torch.count_nonzero(enc.weight) == 0


def test_bandlimited_time_features_stay_below_5hz_nyquist():
    enc=RelativeActionTimeEncoding(
        1024,32,10,min_frequency_hz=0.1,max_frequency_hz=4.0
    )
    frequencies=torch.logspace(
        torch.log10(torch.tensor(0.1)).item(),
        torch.log10(torch.tensor(4.0)).item(),16
    )
    assert frequencies.max() < 5
    assert enc.features(50, torch.device("cpu")).shape == (50,32)
    assert not torch.equal(
        enc.features(50,torch.device("cpu")),
        RelativeActionTimeEncoding(1024,32,10).features(50,torch.device("cpu")),
    )


def test_rms_gate_zero_init_identity_and_per_token_residual():
    torch.manual_seed(2)
    x=torch.randn(2,50,16)
    base=RelativeActionTimeEncoding(16,32,10)
    gated=RelativeActionTimeEncoding(16,32,10,gain_mode="relative_rms")
    torch.testing.assert_close(base(x),x,atol=0,rtol=0)
    torch.testing.assert_close(gated(x),x,atol=0,rtol=0)
    assert sum(q.numel() for q in gated.parameters())==16*32
    assert len(list(gated.parameters()))==1
    with torch.no_grad():
        base.weight.copy_(torch.randn_like(base.weight)*0.01)
        gated.weight.copy_(base.weight)
    x=torch.randn(2,50,16)
    x[:,5,:]*=4
    ungated=base(x)-x
    adapted=gated(x)-x
    assert not torch.equal(adapted,ungated)
    assert adapted[:,5].abs().mean()>ungated[:,5].abs().mean()
    with torch.no_grad():
        # At the same token RMS across the whole chunk, gain must be one.
        constant=torch.ones(2,50,16)
        torch.testing.assert_close(base(constant),gated(constant),atol=0,rtol=0)


def test_rms_gain_gradient_reaches_only_original_32768_parameter_projection():
    e=RelativeActionTimeEncoding(1024,32,10,gain_mode="relative_rms")
    inputs=torch.randn(2,50,1024,requires_grad=True)
    output=e(inputs).square().mean()
    output.backward()
    assert e.weight.grad is not None
    assert torch.isfinite(e.weight.grad).all()
    assert e.weight.grad.abs().sum()>0
    assert sum(p.numel() for p in e.parameters())==32768
    assert inputs.grad is not None


def test_frequency_and_gain_guards_no_legacy_config_breakage():
    assert PI05Config().action_time_max_frequency_hz==10
    assert PI05Config().action_time_gain_mode=="none"
    for invalid in (
        {"action_time_min_frequency_hz":10,"action_time_max_frequency_hz":4},
        {"action_time_min_frequency_hz":float("nan")},
        {"action_time_gain_mode":"oracle_gripper"},
    ):
        with pytest.raises(ValueError):
            PI05Config(**config_params(**invalid))
    with pytest.raises(ValueError,match="requires enabled input injection"):
        PI05Config(action_time_gain_mode="relative_rms")
    with pytest.raises(ValueError,match="requires enabled input injection"):
        PI05Config(**config_params(action_time_injection="qk",action_time_gain_mode="relative_rms"))
    for name,(low,high,mode) in spectra.PARAMETERS.items():
        cfg=PI05Config(**config_params(
            action_time_min_frequency_hz=low,
            action_time_max_frequency_hz=high,
            action_time_gain_mode=mode,
        ))
        assert cfg.action_time_injection=="input"


def model_args(tmp_path):
    return SimpleNamespace(
        directory=tmp_path/"base",
        output_root=tmp_path/"models",
        eval_root=tmp_path/"optimized_eval",
        control_eval_root=tmp_path/"original_eval",
        log_root=tmp_path/"logs",
        tokenizer=tmp_path/"tokenizer",
        dataset_repo_id="libero",
        source=tmp_path/"source",
        gpus="0,1",gpu=0,port=29589,
        num_workers=0,repeat=1,dry_run=True,
        wandb_mode="offline",
        shift_m=.12,shift_seed=20261103,
        episodes=10,
    )


def record():
    return {
        "dataset_root":"/home/wyn/datasets/libero",
        "fps":10.0,
        "settings":{"steps":1000,"batch_size":32,"world_size":2,"data_seed":0},
        "selection":{"task_id":7,"episode_index":7,"chunk_size":50},
    }


def test_original_checkpoints_reused_new_output_prefixes_and_pdet_plan(tmp_path,monkeypatch):
    args=model_args(tmp_path)
    old=spectra.baseline_jobs(record(),args)
    new=spectra.training_jobs(record(),args)
    assert [j["case"] for j in old]==["rope","time_input"]
    assert [j["case"] for j in new]==list(spectra.CASES)
    assert len({j["name"] for j in old+new})==5
    assert all(j["checkpoint"].parts[-2:]==("001000","pretrained_model") for j in old+new)
    assert all(j["plan"]==args.directory/"uniform.npz" for j in new)
    # Never re-enable resampled labels from the experiment-II shell.
    monkeypatch.setenv("PI05_CADENCE_PLAN","/path/should-not-be-used")
    for job in new:
        cmd,env=spectra.training_command(job,record(),args,tmp_path/"prompt.safetensors")
        assert env.get("PI05_CADENCE_PLAN") is None
        assert env["FLOW_STEPS"]=="1000"
        assert env["GPU_IDS"]=="0,1"
        assert "--pdet.flow_binding=occurrence" in cmd
        assert f"--policy.action_time_min_frequency_hz={job['low_hz']}" in cmd
        assert f"--policy.action_time_max_frequency_hz={job['high_hz']}" in cmd
        assert f"--policy.action_time_gain_mode={job['gain_mode']}" in cmd


def test_paired_gpu_scheduler_handles_odd_number_of_models(monkeypatch,tmp_path):
    args=model_args(tmp_path)
    args.record=record()
    calls=[]
    def fake_eval(job,rec,_args,seed,gpu,*,prior=False):
        calls.append((job["case"],seed,gpu,prior))
    monkeypatch.setattr(spectra,"run_eval_job",fake_eval)
    models=[(j,True) for j in spectra.baseline_jobs(record(),args)]
    models +=[(j,False) for j in spectra.training_jobs(record(),args)]
    spectra._run_pair(models,args,5000)
    assert sorted(x[0] for x in calls)==sorted(spectra.ALL_CASES)
    assert sorted(x[2] for x in calls)==[0,0,0,1,1]
    assert sum(x[3] for x in calls)==2
    assert all(x[1]==5000 for x in calls)


def test_summary_excludes_mismatched_physical_shift_and_prints_five_arms(tmp_path,capsys):
    args=model_args(tmp_path)
    rec=record()
    old=spectra.baseline_jobs(rec,args)
    new=spectra.training_jobs(rec,args)
    eval_args=spectra._eval_args(args,5000)
    configs=[spectra._eval_args(args,5000,prior=True),eval_args]
    for cfg in configs:
        root=original.evaluation_root(cfg,rec)
        root.mkdir(parents=True,exist_ok=True)
        shift=original.shift_config(cfg)
        (root/"fixed_object_shift_config.json").write_text(
            json.dumps(shift,sort_keys=True)
        )
    for i,job in enumerate(old+new):
        prior=i<len(old)
        path=spectra._result(job,rec,args,5000,prior=prior)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({
            "per_task":[{
                "task_group":"libero_10","task_id":7,
                "metrics":{"successes":[True]*(6+(i%2)) + [False]*(4-(i%2))}
            }]
        }))
        cfg=spectra._eval_args(args,5000,prior=prior)
        config_path=original.evaluation_config_path(cfg,rec)
        job["checkpoint"].mkdir(parents=True,exist_ok=True)
        model_file=job["checkpoint"]/"model.safetensors"
        model_file.write_bytes(b"fixture")
        files=[[
            str(model_file.relative_to(job["checkpoint"])),
            model_file.stat().st_size,model_file.stat().st_mtime_ns,
        ]]
        (path.parent/"resume_manifest.json").write_text(json.dumps({
            "model_files":files,
            "task7_object_shift":{
                "config": original.shift_config(cfg),
                "config_sha256":previous.file_hash(config_path),
                "helper_sha256":"identical-helper",
            },
        }))
    summary=spectra.summarize_seed(rec,args,5000,old,new)
    assert set(summary["results"])==set(spectra.ALL_CASES)
    assert "lowfreq_rms" in capsys.readouterr().out
    assert all(x["overall_episodes"]==10 for x in summary["results"].values())
    faulty_path=spectra._result(new[-1],rec,args,5000).parent/"resume_manifest.json"
    faulty=json.loads(faulty_path.read_text())
    faulty["task7_object_shift"]["helper_sha256"]="different"
    faulty_path.write_text(json.dumps(faulty))
    with pytest.raises(ValueError,match="physics implementation"):
        spectra.summarize_seed(rec,args,5000,old,new,print_now=False)
    # Tampered model should also invalidate a previously successful eval.
    faulty["task7_object_shift"]["helper_sha256"]="identical-helper"
    faulty_path.write_text(json.dumps(faulty))
    (new[-1]["checkpoint"]/"model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="checkpoint differs"):
        spectra.summarize_seed(rec,args,5000,old,new,print_now=False)


def test_seed_validation_and_no_reusing_calibration_by_default():
    assert spectra.parse_seeds("1000,3000,4000,5000")==[1000,3000,4000,5000]
    for bad in ("","3000,3000","a","-1","3000,"):
        with pytest.raises(ValueError):
            spectra.parse_seeds(bad)


def test_whitelist_and_entry_script_exist():
    root=Path(__file__).resolve().parents[2]
    sh=(root/"examples/training/train_pi05_vlm_only.sh").read_text()
    for item in (
        "--policy.action_time_min_frequency_hz=*",
        "--policy.action_time_max_frequency_hz=*",
        "--policy.action_time_gain_mode=*",
    ):
        assert item in sh
    wrapper=(root/"examples/training/train_pi05_task7_input_spectral.sh").read_text()
    assert "pi05_task7_input_spectrum train" in wrapper
    dual=(root/"run_eval_libero_task7_input_spectral_2gpu.sh").read_text()
    assert "pi05_task7_input_spectrum run" in dual


def test_real_training_shell_accepts_optimised_spectral_gain_flags(tmp_path):
    """Regression: no Unsupported override error, as happened in prior iteration."""
    import os
    import subprocess

    repo=Path(__file__).resolve().parents[2]
    folder=tmp_path/"preflight"
    for name in ("bin","source","dataset","tokenizer"):
        (folder/name).mkdir(parents=True)
    executable=folder/"bin"/"python"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    env={
        **os.environ,
        "PATH":f"{folder / 'bin'}:{os.environ['PATH']}",
        "PRETRAINED_PATH":str(folder/"source"),
        "DATASET_ROOT":str(folder/"dataset"),
        "TOKENIZER_PATH":str(folder/"tokenizer"),
        "OUTPUT_ROOT":str(folder/"models"),
        "RUN_NAME":"spectral-shell-preflight",
        "GPU_IDS":"0,1","NUM_PROCESSES":"2",
        "BATCH_SIZE":"32","FLOW_STEPS":"1000",
        "TRAIN_ACTION_EXPERT":"true","VLM_PROMPT_TOKENS":"2",
        "SAVE_ONLY_FINAL":"true","DRY_RUN":"true",
    }
    cmd=[
        "bash",str(repo/"examples/training/train_pi05_vlm_only.sh"),
        "base","0","--policy.compile_model=false",
        "--policy.action_time_encoding=true",
        "--policy.action_time_min_frequency_hz=0.1",
        "--policy.action_time_max_frequency_hz=4.0",
        "--policy.action_time_gain_mode=relative_rms",
    ]
    result=subprocess.run(cmd,env=env,cwd=repo,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    for flag in cmd[4:]:
        assert flag in result.stdout
    assert "--steps=1000" in result.stdout


def test_new_shift_seed_configs_precreated_before_gpu_parallelism(tmp_path):
    args=model_args(tmp_path)
    rec=record()
    spectra.prepare_shift_configs(rec,args,6000)
    for prior in (False,True):
        cfg=spectra._eval_args(args,6000,prior=prior)
        saved=original.evaluation_config_path(cfg,rec)
        assert saved.is_file()
        assert json.loads(saved.read_text()) == original.shift_config(cfg)
    spectra.prepare_shift_configs(rec,args,6000)
