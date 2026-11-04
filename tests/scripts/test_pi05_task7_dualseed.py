"""CPU-only guards for independent scene/policy RNG and baseline-difficulty lock."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.scripts import pi05_task7_object_shift as legacy
from lerobot.scripts import pi05_task7_dualseed_shift as shift
from lerobot.scripts import pi05_task7_dualseed_eval as study
from lerobot.scripts import pi05_task7_timepath_experiment as original
from lerobot.scripts import pi05_task7_input_spectrum as spectral


def cfg(scene=101,policy=1,init=3000,m=0.06):
    return {
        "protocol":shift.PROTOCOL,"task_id":7,"suite":"libero_10",
        "shift_m":m,"scene_seed":scene,"policy_seed":policy,"eval_seed":init,
    }


def positions():
    def p(x,y):
        return np.asarray([x,y,.82,1.,0.,0.,0.])
    return {
        "alphabet_soup_1":p(.05,-.10),
        "cream_cheese_1":p(-.15,.06),
        "tomato_sauce_1":p(.1,-.2),
        "ketchup_1":p(-.2,-.15),
        "basket_1":p(0.,.26),
    }


def temp_args(tmp_path,scene=101,policy=1):
    return SimpleNamespace(
        shift_m=.12,shift_seed=20261103,scene_seed=scene,
        policy_seed=policy,eval_seed=3000,episodes=10,repeat=1,
        eval_root=tmp_path,
    )


def test_v2_scene_changes_only_with_scene_seed_not_policy_seed():
    xyz=positions()
    a=shift.choose_translations(xyz,cfg(scene=100,policy=30),3001)
    b=shift.choose_translations(xyz,cfg(scene=100,policy=31),3001)
    c=shift.choose_translations(xyz,cfg(scene=101,policy=30),3001)
    for key in legacy.TARGET_OBJECTS:
        np.testing.assert_array_equal(a[key],b[key])
        assert not np.array_equal(a[key][:2],c[key][:2])
        assert np.linalg.norm(a[key][:2]-xyz[key][:2])==pytest.approx(.06)
    assert shift.policy_rng_seed(cfg(scene=100,policy=31),1)!=shift.policy_rng_seed(
        cfg(scene=100,policy=30),1
    )
    assert shift.policy_rng_seed(cfg(scene=100,policy=30),1)==shift.policy_rng_seed(
        cfg(scene=101,policy=30),1
    )


def test_v2_scene_indices_do_not_depend_on_absolute_eval_seed():
    xyz=positions()
    a=shift.choose_translations(xyz,cfg(scene=111,policy=4,init=3000),3007)
    b=shift.choose_translations(xyz,cfg(scene=111,policy=4,init=9000),9007)
    for key in legacy.TARGET_OBJECTS:
        np.testing.assert_array_equal(a[key],b[key])
    assert shift.init_state_for_episode(3007,cfg(),10)==7
    assert shift.init_state_for_episode(3007,cfg(),50)==7
    with pytest.raises(ValueError,match="exactly 10"):
        shift.init_state_for_episode(3010,cfg(),10)
    with pytest.raises(ValueError,match="at least 10"):
        shift.init_state_for_episode(3000,cfg(),9)


def test_legacy_v1_config_and_randomness_unchanged():
    old={"protocol":legacy.PROTOCOL,"task_id":7,"suite":"libero_10",
         "shift_m":.06,"seed":20261103,"eval_seed":3000}
    assert legacy.validate_config(old)==old
    same=legacy.choose_translations(positions(),old,3001)
    for name,p in same.items():
        np.testing.assert_array_equal(p,legacy.choose_translations(positions(),old,3001)[name])
    with pytest.raises(ValueError):
        legacy.validate_config(cfg())
    with pytest.raises(ValueError):
        shift.validate_config(old)


def test_v2_policy_rng_reproducible_with_evaluator_hook(tmp_path):
    cfg_path=tmp_path/"shift.json"
    cfg_path.write_text(json.dumps(cfg(scene=101,policy=222)))
    evaluator=SimpleNamespace()
    evaluator.rollout=lambda *,seeds: torch.randn(7)
    evaluator.run_one=lambda task_group,task_id,env,*,start_seed: evaluator.rollout(seeds=[start_seed])
    initial=evaluator.rollout
    trace=shift.install(evaluator,cfg_path)
    assert trace["task7_dualseed_shift"]["config"]==cfg(scene=101,policy=222)
    first=evaluator.run_one("libero_10",7,None,start_seed=3003)
    torch.randn(100)
    second=evaluator.run_one("libero_10",7,None,start_seed=3003)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert not torch.equal(first,evaluator.run_one("libero_10",7,None,start_seed=3004))
    with pytest.raises(ValueError,match="only run standard task7"):
        evaluator.run_one("libero_10",8,None,start_seed=3001)


def test_v2_eval_paths_isolated_from_legacy_v1_results(tmp_path):
    record={"selection":{"task_id":7,"episode_index":7}}
    a=temp_args(tmp_path,scene=101,policy=3)
    b=temp_args(tmp_path,scene=101,policy=4)
    c=temp_args(tmp_path,scene=102,policy=3)
    old=SimpleNamespace(**{**vars(a),"scene_seed":None,"policy_seed":None})
    old_cfg=original.shift_config(old)
    assert old_cfg["protocol"]==legacy.PROTOCOL
    assert original.shift_config(a)==cfg(scene=101,policy=3,init=3000,m=.12)
    old_path=original.evaluation_root(old,record)
    assert "fixed-object-shift-dualseed" not in str(old_path)
    assert "fixed-object-shift-dualseed" in str(original.evaluation_root(a,record))
    assert original.evaluation_root(a,record)!=original.evaluation_root(b,record)
    assert original.evaluation_root(a,record)!=original.evaluation_root(c,record)
    with pytest.raises(ValueError,match="both"):
        original.shift_config(SimpleNamespace(**{**vars(a),"policy_seed":None}))


def test_spectral_eval_context_preserves_independent_seeds(tmp_path):
    args=SimpleNamespace(
        scene_seed=101,policy_seed=303,shift_m=.12,shift_seed=20261103,
        repeat=1,eval_root=tmp_path/"new",
        control_eval_root=tmp_path/"old",
    )
    old=spectral._eval_args(args,3000,prior=True)
    new=spectral._eval_args(args,3000,prior=False)
    assert old.scene_seed==new.scene_seed==101
    assert old.policy_seed==new.policy_seed==303
    assert old.eval_root!=new.eval_root
    assert original.shift_config(old)==original.shift_config(new)


def test_seed_lists_must_be_disjoint_by_protocol():
    assert study.seeds("101,102","test")== (101,102)
    for x in ("","a","1,1","-1,4","1,,3"):
        with pytest.raises(ValueError):
            study.seeds(x,"test")
    assert study.case_names("rope,time_input,lowfreq")==("rope","time_input","lowfreq")
    assert len(study.case_names("all"))==7
    with pytest.raises(ValueError,match="baseline"):
        study.case_names("time_input,lowfreq")


def test_rope_calibration_target_is_pooled_not_selectively_subsampled():
    rows=[
        {"successes":2,"episodes":10},
        {"successes":7,"episodes":10},
        {"successes":4,"episodes":10},
        {"successes":5,"episodes":10},
    ]
    outcome=study.summarize_calibration(rows,40,50)
    assert outcome["successes"]==18
    assert outcome["episodes"]==40
    assert outcome["rate_percent"]==45
    assert outcome["in_target_band"]
    assert not study.summarize_calibration(
        [{"successes":7,"episodes":10}],40,50,
    )["in_target_band"]


def test_paired_count_explains_net_success_gain():
    a={"per_episode":[True,True,False,False]}
    b={"per_episode":[True,False,True,True]}
    result=study.paired_statistics(a,b)
    assert result=={"model_only_wins":2,"rope_only_wins":1,"net_wins":1}


def test_dryrun_never_starts_gpu_or_uses_previous_results(tmp_path):
    import os
    import subprocess
    import sys

    repo=Path(__file__).resolve().parents[2]
    env={**os.environ,"PYTHONPATH":str(repo/"src")+":"+os.environ.get("PYTHONPATH","")}
    command=[
        sys.executable,"-m","lerobot.scripts.pi05_task7_dualseed_eval",
        "calibrate","--dry-run","--scene-seeds","111,112",
        "--cal-policy-seeds","8,9","--test-policy-seeds","10,11",
        "--report-root",str(tmp_path/"reports"),
    ]
    proc=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert proc.stdout.count("CAL:")==4
    assert "no GPU allocations" in proc.stdout
    assert not (tmp_path/"reports").exists()


def test_shell_dry_run_and_legacy_evaluator_compatibility():
    root=Path(__file__).resolve().parents[2]
    script=(root/"run_eval_libero_task7_dualseed_2gpu.sh").read_text()
    assert "pi05_task7_dualseed_eval" in script
    base=(root/"run_eval_libero-full.sh").read_text()
    assert "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG" in base
    assert "LIBERO_TASK7_OBJECT_SHIFT_CONFIG" in base
    assert "task7_dualseed_shift" in (root/"src/lerobot/scripts/pi05_task7_dualseed_shift.py").read_text()
    env=(root/"src/lerobot/envs/libero.py").read_text()
    assert "dual_path and legacy_path" in env


def _study_fixture(tmp_path):
    checkpoint=tmp_path/"models"/"rope"/"checkpoints"/"001000"/"pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint/"model.safetensors").write_bytes(b"fixture")
    directory=tmp_path/"plan"
    directory.mkdir()
    (directory/"manifest.json").write_text("{}")
    args=SimpleNamespace(
        scene_seeds=(101,102),cal_policy_seeds=(7,8),test_policy_seeds=(701,702),
        external_scene_seeds=(),target_min=40.,target_max=50.,shift_m=.12,
        init_seed=3000,repeat=1,gpus="0,1",
        directory=directory,output_root=tmp_path/"models",tokenizer=tmp_path/"tokenizer",
        control_eval_root=tmp_path/"control_eval",optimized_eval_root=tmp_path/"optim_eval",
        report_root=tmp_path/"reports",lock_file=tmp_path/"reports"/"calibration-lock.json",
        log_dir=tmp_path/"logs",cases="rope,time_input",
    )
    record={"selection":{"task_id":7,"episode_index":7}}
    return args,record,{"rope":{"checkpoint":checkpoint,"case":"rope"}}


def test_calibration_writes_lock_only_on_40_to_50_percent_pooled(tmp_path,monkeypatch):
    args,record,models=_study_fixture(tmp_path)
    rates={(101,7):2,(101,8):7,(102,7):4,(102,8):5}
    monkeypatch.setattr(study,"precreate_configs",lambda *a:None)
    def fake_one(_record,_args,job,scene,policy,gpu):
        successes=rates[(scene,policy)]
        return {
            "scene_seed":scene,"policy_seed":policy,"case":"rope",
            "successes":successes,"episodes":10,
            "per_episode":[True]*successes+[False]*(10-successes),
            "eval_info":f"fake-{scene}-{policy}","checkpoint":str(job["checkpoint"]),
        }
    monkeypatch.setattr(study,"evaluate_one",fake_one)
    monkeypatch.setattr(
        study,"verified_result",
        lambda rec,params,job,scene,policy:fake_one(rec,params,job,scene,policy,0)
    )
    outcome=study.calibrate(record,args,models)
    assert outcome["calibration"]["rate_percent"]==45
    assert args.lock_file.exists()
    assert len(outcome["calibration_rows"])==4
    assert study.verify_lock(args,record,models)==outcome
    # No changing the heldout-policy schedule after looking at calibration.
    args.test_policy_seeds=(703,704)
    with pytest.raises(ValueError,match="changed"):
        study.verify_lock(args,record,models)


def test_out_of_band_calibration_does_not_lock_or_relabel_seed(tmp_path,monkeypatch):
    args,record,models=_study_fixture(tmp_path)
    monkeypatch.setattr(study,"precreate_configs",lambda *a:None)
    monkeypatch.setattr(
        study,"evaluate_one",
        lambda rec,params,job,scene,policy,gpu:{
            "scene_seed":scene,"policy_seed":policy,"case":"rope",
            "successes":7,"episodes":10,
            "per_episode":[True]*7+[False]*3,
            "eval_info":f"fake-{scene}-{policy}","checkpoint":str(job["checkpoint"]),
        }
    )
    with pytest.raises(ValueError,match="outside"):
        study.calibrate(record,args,models)
    assert not args.lock_file.exists()
    assert len(list(args.report_root.glob("calibration-*.json")))==1
    saved=json.loads(next(args.report_root.glob("calibration-*.json")).read_text())
    assert not saved["calibration"]["in_target_band"]
    assert saved["calibration"]["rate_percent"]==70


def test_validation_command_explicitly_locks_scene_and_policy(tmp_path):
    args,record,models=_study_fixture(tmp_path)
    cmd=study.evaluation_command(record,args,models["rope"],101,701,0)
    assert "--scene-seed" in cmd
    assert "--policy-seed" in cmd
    assert cmd[cmd.index("--scene-seed")+1]=="101"
    assert cmd[cmd.index("--policy-seed")+1]=="701"
    assert cmd[cmd.index("--eval-seed")+1]=="3000"
    assert cmd[-2:]==["--gpu","0"]
