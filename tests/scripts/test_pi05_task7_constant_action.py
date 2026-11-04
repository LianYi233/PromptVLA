"""Constant 50-action visual probe must isolate the learned input-time path."""

import json

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.scripts import pi05_task7_constant_action as probe


def _model(*, gain_mode="none"):
    encoder = RelativeActionTimeEncoding(24, 32, 10.0, gain_mode=gain_mode)
    generator = torch.Generator().manual_seed(71)
    with torch.no_grad():
        encoder.weight.copy_(torch.randn(24, 32, generator=generator) * 0.03)
    projection = torch.randn(24, 32, generator=generator) * 0.1
    bias = torch.randn(24, generator=generator) * 0.01
    action = probe.parse_action(probe.DEFAULT_ACTION)
    return encoder, projection, bias, action


@pytest.mark.parametrize("gain_mode", ["none", "relative_rms"])
def test_constant_actions_only_differ_after_input_time_encoding(gain_mode):
    encoder, projection, bias, action = _model(gain_mode=gain_mode)
    data = probe.constant_action_probe(encoder, projection, bias, action)
    assert data["action"].shape == (50, 7)
    assert data["rope_input"].shape == (50, 24)
    assert data["time_features"].shape == (50, 32)
    assert np.array_equal(data["action"], np.repeat(data["action"][:1], 50, axis=0))
    assert np.array_equal(data["rope_input"], np.repeat(data["rope_input"][:1], 50, axis=0))
    np.testing.assert_array_equal(data["zero"], data["rope_input"])
    np.testing.assert_allclose(data["correct"], data["reversed"][::-1], atol=1e-6, rtol=0)
    summary = probe.summarize(data)
    assert summary["all_50_actions_identical"]
    assert summary["rope_only_inputs_identical"]
    assert summary["zero_equals_rope_input"]
    assert summary["position_signal_detected_at_input"]
    assert summary["correct_max_l2_difference_from_first"] > 0.001
    assert summary["reversed_vs_correct_rms"] > 0
    assert summary["correct_vs_zero_rms"] > 0


def test_untrained_zero_projection_does_not_fabricate_temporal_separation():
    encoder, projection, bias, action = _model()
    with torch.no_grad():
        encoder.weight.zero_()
    data = probe.constant_action_probe(encoder, projection, bias, action)
    np.testing.assert_array_equal(data["correct"], data["rope_input"])
    np.testing.assert_array_equal(data["reversed"], data["rope_input"])
    assert not probe.summarize(data)["position_signal_detected_at_input"]


def test_explicit_physical_dt_and_global_dt_do_not_get_confused_with_flow_time():
    encoder, projection, bias, action = _model()
    physical = probe.constant_action_probe(
        encoder, projection, bias, action, basis="physical", delta_t=0.2
    )
    assert physical["time_seconds"][1] == pytest.approx(0.2)
    assert physical["time_seconds"][-1] == pytest.approx(9.8)
    assert probe.summarize(physical)["position_signal_detected_at_input"]
    global_dt = probe.constant_action_probe(
        encoder, projection, bias, action, basis="global_dt", delta_t=0.2
    )
    assert np.allclose(global_dt["time_seconds"], 0.2)
    # Global-dt encodes cadence, not which of the 50 positions is which.
    assert not probe.summarize(global_dt)["position_signal_detected_at_input"]
    with pytest.raises(ValueError, match="only applies"):
        probe.constant_action_probe(encoder, projection, bias, action, delta_t=0.1)


@pytest.mark.parametrize("bad", ["1,2,3", "1,2,3,4,5,6,nan", "1,2,3,4,5,6,inf"])
def test_reject_malformed_action_vector(bad):
    with pytest.raises(ValueError, match="seven"):
        probe.parse_action(bad)


def test_checkpoint_loads_real_weights_without_constructing_vla(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    encoder, projection, bias, action = _model()
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config = {
        "chunk_size": 50,
        "max_action_dim": 32,
        "action_time_encoding": True,
        "action_time_injection": "input",
        "action_time_encoding_fps": 10.0,
        "action_time_encoding_dim": 32,
        "action_time_basis": "index",
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    safetensors.save_file({
        "model.action_in_proj.weight": projection,
        "model.action_in_proj.bias": bias,
        "model.action_time_encoder.weight": encoder.weight.detach().clone(),
    }, str(checkpoint / "model.safetensors"))
    loaded, weight, loaded_bias, basis, source = probe.load_checkpoint(checkpoint)
    assert basis == "index"
    assert source["kind"] == "trained_checkpoint"
    torch.testing.assert_close(weight, projection)
    torch.testing.assert_close(loaded_bias, bias)
    torch.testing.assert_close(loaded.weight, encoder.weight)
    assert probe.summarize(probe.constant_action_probe(
        loaded, weight, loaded_bias, action
    ))["position_signal_detected_at_input"]

    config["action_time_injection"] = "qk"
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="action_time_injection=input"):
        probe.load_checkpoint(checkpoint)


def test_figure_and_machine_readable_data_are_generated_from_actual_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    encoder, projection, bias, action = _model()
    data = probe.constant_action_probe(encoder, projection, bias, action)
    source = {"kind": "illustrative_untrained"}
    report_path = probe.save_results(tmp_path, data, probe.summarize(data), source,
                                     encoder, "index")
    report = json.loads(report_path.read_text())
    assert report["source"]["kind"] == "illustrative_untrained"
    assert report["metrics"]["all_50_actions_identical"]
    assert len(report["figures"]) == 2
    assert all((tmp_path / path.split("/")[-1]).stat().st_size > 500
               for path in report["figures"])
    exported = np.load(tmp_path / "constant_action_50_embeddings.npz")
    np.testing.assert_array_equal(exported["action"], data["action"])
    csv = np.loadtxt(tmp_path / "constant_action_50_positions.csv",
                     delimiter=",", skiprows=1)
    assert csv.shape == (50, 5)
    assert np.max(csv[:, 3]) == pytest.approx(
        report["metrics"]["correct_max_l2_difference_from_first"], rel=1e-6
    )
