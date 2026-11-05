"""Saved comparisons must keep scene/policy/init identities and all outcomes."""

import copy
import json

import pytest

from lerobot.scripts import pi05_task7_dualseed_evidence as evidence


def comparison():
    cells = []
    cases = ["rope", "time_input", "lowfreq", "rms_gate", "lowfreq_rms"]
    for scene in [11, 12]:
        for policy in [101, 102, 103]:
            results = {}
            for case in cases:
                flags = [True, False, False, True, False, False, False, False, False, False]
                if case == "time_input":
                    flags[1] = True
                    if scene == 12:
                        flags[3] = False
                results[case] = {"case": case, "scene_seed": scene, "policy_seed": policy,
                                 "per_episode": flags, "successes": sum(flags), "episodes": 10,
                                 "checkpoint": f"/models/{case}",
                                 "eval_info": f"/eval/{scene}/{policy}/{case}/eval_info.json"}
            cells.append({"scene_seed": scene, "policy_seed": policy, "results": results})
    return {"protocol": evidence.FIXED_PROTOCOL, "stage": "fixed_protocol", "cases": cases,
            "settings": {"scene_seeds": [11, 12], "test_policy_seeds": [101, 102, 103],
                         "fixed_init_states": list(range(10))}, "evaluations": cells}


def test_complete_five_method_grid_preserves_scene_difference():
    result = evidence.validate_comparison(comparison())
    assert result["aggregate"]["rope"] == [12, 60]
    assert result["aggregate"]["time_input"] == [15, 60]
    assert result["paired_counts"]["time_input"]["model_only_wins"] == 6
    assert result["paired_counts"]["time_input"]["rope_only_wins"] == 3
    assert result["by_scene"]["11"]["paired_counts"]["time_input"]["net_wins"] == 3
    assert result["by_scene"]["12"]["paired_counts"]["time_input"]["net_wins"] == 0
    assert len(result["time_input_discordant_episodes"]) == 9


@pytest.mark.parametrize("problem", ["missing", "duplicate", "identity", "count", "checkpoint", "summary"])
def test_rejects_incomplete_or_mispaired_evidence(problem):
    report = comparison()
    if problem == "missing":
        report["evaluations"].pop()
    elif problem == "duplicate":
        report["evaluations"].append(copy.deepcopy(report["evaluations"][0]))
    elif problem == "identity":
        report["evaluations"][0]["results"]["time_input"]["policy_seed"] = 104
    elif problem == "count":
        report["evaluations"][0]["results"]["time_input"]["successes"] = 10
    elif problem == "checkpoint":
        report["evaluations"][1]["results"]["time_input"]["checkpoint"] = "/other/checkpoint"
    else:
        report["summary"] = {"aggregate": {"rope": [99, 60]}}
    with pytest.raises(ValueError):
        evidence.validate_comparison(report)


def test_original_calibration_report_supported_without_pooling_calibration_draws():
    report = comparison()
    report["protocol"] = evidence.DUAL_PROTOCOL
    report["stage"] = "calibrated_stress"
    report["lock"] = {"settings": {"protocol": evidence.DUAL_PROTOCOL,
                                    "scene_seeds": [11, 12],
                                    "heldout_policy_seeds": [101, 102, 103],
                                    "calibration_policy_seeds": [1, 2]},
                      "calibration": {"in_target_band": True},
                      "calibration_rows": [{"successes": 999}]}
    assert evidence.validate_comparison(report)["aggregate"]["rope"] == [12, 60]
    report["lock"]["settings"]["calibration_policy_seeds"] = [101]
    with pytest.raises(ValueError, match="overlap"):
        evidence.validate_comparison(report)


def test_two_model_pipeline_and_real_data_driven_plot(tmp_path):
    pytest.importorskip("matplotlib")
    report = comparison()
    report["cases"] = ["rope", "time_input"]
    for cell in report["evaluations"]:
        cell["results"] = {k: v for k, v in cell["results"].items() if k in report["cases"]}
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(report))
    result = evidence.run_success_report(source, tmp_path / "figures")
    assert result["stage"] == "fixed_protocol"
    assert len(result["source_report_sha256"]) == 64
    assert (tmp_path / "figures" / "fig01_paired_task7_outcomes.png").stat().st_size > 1000
    assert (tmp_path / "figures" / "fig01_paired_task7_outcomes.svg").stat().st_size > 1000
    saved = json.loads((tmp_path / "figures" / "paired_success.json").read_text())
    assert saved["aggregate"]["time_input"] == [15, 60]
