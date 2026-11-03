"""CPU-only tests for the unified eight-policy, two-metric task7 reporter."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.scripts import pi05_task7_joint_eval as joint


def settings(tmp_path):
    return SimpleNamespace(
        only="all", repeat=1, episodes=10,
        timepath_seeds=[1000,3000,4000,5000], cadence_seed=3000,
        shift_m=0.12, shift_seed=20261103, flow_time=0.5,
        max_eval_starts=32, dataset_repo_id="libero",
        tokenizer=tmp_path/"tokenizer", log_root=tmp_path/"logs",
        output_root=tmp_path/"models", report_dir=tmp_path/"joint",
        timepath_directory=tmp_path/"old_plan",
        cadence_directory=tmp_path/"cadence_plan",
        timepath_eval_root=tmp_path/"model_eval",
        cadence_eval_root=tmp_path/"cadence_eval",
        gpus="0,1",
    )


def fake_successes(hits):
    return {
        "overall_successes": hits,
        "overall_episodes": 10,
        "selected_task_rate": 10.0*hits,
    }


def qkv_results():
    return {
        seed: {
            "episodes": 10,
            "results": {
                "rope": fake_successes(4 if seed != 3000 else 6),
                "time_qk": fake_successes(5 if seed != 3000 else 6),
                "time_v": fake_successes(5 if seed != 3000 else 7),
                "time_input": fake_successes(6 if seed != 3000 else 7),
            },
        }
        for seed in (1000,3000,4000,5000)
    }


def fake_stride(mse, starts=(1,6,11)):
    return {
        "mean_mse": mse,
        "sample_mse": [mse for _ in starts],
        "n": len(starts),
        "start_indices": list(starts),
    }


def cadence_results():
    return {
        "results": {
            case: {
                "1": fake_stride(1.3-0.1*i),
                "2": fake_stride(1.7-0.2*i),
                "3": fake_stride(2.0-0.15*i, starts=(1,6)),
            }
            for i,case in enumerate(joint.CASES_II)
        },
    }


def test_seed_selection_and_calibration_not_added_by_default():
    assert joint.parse_seeds("1000,3000,4000,5000") == [1000,3000,4000,5000]
    for bad in ("", "a", "0,0", "-2,1", "1,,2"):
        with pytest.raises(ValueError):
            joint.parse_seeds(bad)


def test_two_existing_two_gpu_launchers_receive_only_their_own_options(tmp_path):
    args = settings(tmp_path)
    c1 = joint.pair_command(args, "timepath", seed=3000)
    c2 = joint.pair_command(args, "cadence")
    assert c1[0] == c2[0] == "bash"
    assert c1[1].endswith("run_eval_libero_task7_timepath_2gpu.sh")
    assert c2[1].endswith("run_eval_libero_task7_physical_time_2gpu.sh")
    assert c1[c1.index("--eval-seed") + 1] == "3000"
    assert c2[c2.index("--eval-seed") + 1] == "3000"
    assert c1[c1.index("--shift-m") + 1] == "0.12"
    assert "--shift-m" not in c2
    assert c2[c2.index("--flow-time") + 1] == "0.5"
    assert "--flow-time" not in c1
    assert c1[c1.index("--episodes") + 1] == "10"
    assert "--episodes" not in c2
    assert joint.selected(args, "timepath") and joint.selected(args, "cadence")


def test_qkv_rates_validated_and_pooled_by_counts_not_seed_averaging(capsys,tmp_path):
    args=settings(tmp_path)
    data=qkv_results()
    joint._check_timepath_rates(data,10)
    joint.print_report(data,None,args)
    out=capsys.readouterr().out
    assert "[Experiment I]" in out
    assert "POOLED" in out
    assert "21/40 (52.5%)" in out
    assert "25/40 (62.5%)" in out
    assert "[Experiment II]" not in out
    wrong=copy.deepcopy(data)
    wrong[3000]["results"]["rope"]["selected_task_rate"]=80
    with pytest.raises(ValueError, match="disagreement"):
        joint._check_timepath_rates(wrong,10)


def test_cadence_mse_all_four_arms_same_heldout_starts(capsys,tmp_path):
    args=settings(tmp_path)
    data=cadence_results()
    joint._check_cadence_paired(data["results"])
    joint.print_report({},data,args)
    out=capsys.readouterr().out
    assert "[Experiment II]" in out
    assert "Global-Δt" in out
    assert "1x" in out and "2x" in out and "3x" in out
    assert "[Experiment I]" not in out
    wrong=copy.deepcopy(data)
    wrong["results"]["index"]["2"]["start_indices"][0]=999
    with pytest.raises(ValueError, match="unpaired"):
        joint._check_cadence_paired(wrong["results"])


def test_summary_persists_separate_metrics_and_never_pools_mse_with_success(tmp_path):
    args=settings(tmp_path)
    t=qkv_results()
    c=cadence_results()
    joint.write_report(t,c,args)
    report=list(args.report_dir.glob("eightway-*.json"))
    assert len(report)==1
    import json

    saved=json.loads(report[0].read_text())
    assert saved["protocol"]==joint.PROTOCOL
    assert len(saved["timepath"])==4
    assert set(saved["cadence"]["results"])==set(joint.CASES_II)
    assert "incomparable" in saved["warning"]
    saved_csv=list(args.report_dir.glob("*.csv"))
    assert len(saved_csv)==2
    first=next(p for p in saved_csv if p.name.endswith("-timepath.csv"))
    second=next(p for p in saved_csv if p.name.endswith("-cadence.csv"))
    assert "successes" in first.read_text()
    assert "flow_mse" in second.read_text()


def test_invalid_strides_are_rejected():
    missing=cadence_results()["results"]
    missing["rope"].pop("2")
    with pytest.raises(ValueError, match="both trained cadences"):
        joint._check_cadence_paired(missing)
    wrong=cadence_results()["results"]
    wrong["global_dt"]["3"]["sample_mse"]=[float("nan")]*2
    with pytest.raises(ValueError, match="unpaired"):
        joint._check_cadence_paired(wrong)


def test_cli_dry_run_no_dataset_load_or_gpu(tmp_path):
    import os
    import subprocess
    import sys

    repo=Path(__file__).resolve().parents[2]
    env={**os.environ, "GPU_IDS":"0,1",
         "PYTHONPATH":f"{repo / 'src'}:{os.environ.get('PYTHONPATH','')}"}
    proc=subprocess.run(
        [
            sys.executable, "-m", "lerobot.scripts.pi05_task7_joint_eval",
            "run", "--dry-run", "--timepath-seeds", "3000",
            "--output-root", str(tmp_path/"outputs"),
        ],
        cwd=repo, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode==0, proc.stderr
    assert proc.stdout.count("run_eval_libero_task7_timepath_2gpu.sh")==1
    assert proc.stdout.count("run_eval_libero_task7_physical_time_2gpu.sh")==1
    assert "no data loaded or GPUs used" in proc.stdout


def test_wrapper_exposes_new_unified_cli_and_dry_run():
    root=Path(__file__).resolve().parents[2]
    sh=(root/"run_eval_libero_task7_all8_2gpu.sh").read_text()
    assert "pi05_task7_joint_eval" in sh
    assert 'DRY_RUN:-false' in sh
