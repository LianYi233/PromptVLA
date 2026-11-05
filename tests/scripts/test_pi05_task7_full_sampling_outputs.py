"""Synthetic run/artifact fixtures; these never measure a real robot policy.

Checkpoint, dataset and model loading are replaced. The real condition loop,
paired aggregation, report/array writing and Matplotlib renderer are exercised.
Native sampler correctness is covered separately in test_pi05_task7_full_sampling.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.scripts import pi05_task7_full_sampling as sampling
from lerobot.scripts import pi05_task7_time_evidence as evidence
from lerobot.scripts import pi05_task7_timepath_experiment as timepath


def _synthetic_target(frame):
    """Deterministic fixture labels, not predictions from a trained checkpoint."""
    local = frame % 250 + np.arange(50, dtype=np.float32)
    demo = frame // 250
    target = np.empty((50, 7), dtype=np.float32)
    for dim in range(6):
        target[:, dim] = 0.2 * np.sin(local * 0.025 + dim * 0.4) + demo * 0.01
    target[:, 6] = np.where(local < 118 + demo * 2, 1.0, -1.0)
    return target


def _fixture_run(monkeypatch, root, *, outcome="zero", explicit_checkpoint=True, visual=False):
    """Return a run configuration and auditable lightweight external dependencies."""
    root = Path(root)
    checkpoint = root / "synthetic-checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    configuration = {
        "chunk_size": 50, "action_time_encoding": True,
        "action_time_injection": "input", "action_time_basis": "index",
        "action_time_encoding_fps": 10.0, "training_stage": "flow",
        "num_inference_steps": 4, "max_action_dim": 32,
        "output_features": {"action": {"shape": [7]}},
        "n_action_steps": 50, "compile_model": False, "rtc_config": None,
    }
    (checkpoint / "config.json").write_text(json.dumps(configuration), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"SYNTHETIC TEST FIXTURE; not real model weights")
    args = sampling.make_parser().parse_args([
        "--directory", str(root / "synthetic-plan"),
        "--output-root", str(root / "synthetic-runs"),
        "--output", str(root / "outputs"), "--repeat", "2", "--allow-cpu",
        "--tokenizer", str(root / "synthetic-tokenizer"),
    ])
    args.noise_seeds = (3000, 4000)
    args.time_input_checkpoint = checkpoint if explicit_checkpoint else None
    record = {
        "dataset_root": "/synthetic/test-fixture/libero",
        "fps": 10.0, "selection": {"episode_index": 7},
    }
    demo_ids = [9, 74, 135, 174, 208, 250, 286, 373] if visual else [9, 74]
    metadata = {
        "episode_index": np.repeat(demo_ids, 250),
        "frame_index": np.tile(np.arange(250), len(demo_ids)),
    }
    # Unequal window counts verify that repeated windows do not overweight a demo.
    anchors = ([(demo, index * 250 + start) for index, demo in enumerate(demo_ids)
                for start in (0, 100, 200)] if visual else [(9, 0), (9, 100), (74, 350)])
    policy = SimpleNamespace(
        config=SimpleNamespace(**configuration),
        model=SimpleNamespace(action_time_encoder=SimpleNamespace(
            weight=torch.arange(16, dtype=torch.float32).reshape(4, 4) / 100.0,
        )),
    )
    fixture = SimpleNamespace(
        calls=[], loads=[], jobs=[], anchors=anchors, checkpoint=checkpoint,
        policy=policy, record=record, metadata=metadata,
    )

    def plan(directory):
        assert directory == args.directory
        return record

    def jobs(saved_record, directory, output_root, *, selection, repeat):
        assert saved_record == record and directory == args.directory
        assert output_root == args.output_root
        assert selection == "time_input" and repeat == 2
        fixture.jobs.append((selection, repeat))
        return [{"checkpoint": checkpoint}]

    def provenance(checkpoints, saved_record):
        assert checkpoints == {"time_input": checkpoint.resolve()}
        assert saved_record == record
        return {"fixture": True, "description": "Synthetic model loading for tests only"}

    def probe_data(received, saved_record):
        assert received is args and saved_record == record
        return anchors, object(), np.zeros((250 * len(demo_ids), 7)), metadata

    def load_model(path, tokenizer, device):
        assert path == checkpoint.resolve() and tokenizer == args.tokenizer
        assert str(device) == "cpu"
        fixture.loads.append(path)
        return policy, object()

    def batches(dataset, actions, frame, processor):
        assert frame in {index for _, index in anchors}
        return {
            "observation.fixture_frame": torch.tensor([[frame]]),
            "task": ["Synthetic artifact-test fixture only"],
        }, _synthetic_target(frame)

    def sample(loaded, inputs, noise, mode):
        assert loaded is policy
        assert set(inputs) == {"observation.fixture_frame", "task"}
        assert noise.shape == (1, 50, 32)
        frame = int(inputs["observation.fixture_frame"].item())
        fixture.calls.append({"frame": frame, "mode": mode, "noise": noise.clone()})
        generated = torch.from_numpy(_synthetic_target(frame)).unsqueeze(0).clone()
        generated[:, :, :6] += 0.02 * torch.tanh(noise[:, :, :6])
        if outcome == "zero":
            offset = 0.2
        elif outcome == "adverse":
            # Every arm component in Correct has larger error than either intervention.
            offset = {"correct": 0.3, "zero": 0.1, "reversed": 0.2}[mode]
        elif outcome == "mixed":
            offset = 0.15
            if mode != "correct":
                positions = torch.arange(50, dtype=torch.float32)
                change = 0.06 * torch.sin(positions * 0.18 + frame // 250)
                if mode == "reversed":
                    change = change * -1.3
                generated[:, :, :6] += change[None, :, None]
        else:
            raise AssertionError(outcome)
        generated[:, :, :6] += offset
        # A shared synthetic event delay gives the renderer a visible target contrast.
        local = frame % 250 + torch.arange(50)
        generated[0, :, 6] = torch.where(local < 125 + (frame // 250) * 2, 1.0, -1.0)
        generated[:, :, 6] += 0.02 * torch.tanh(noise[:, :, 6])
        return generated, {
            "intervention": mode, "initial_noise_sha256": sampling.tensor_sha256(noise),
            "observation_sha256": sampling.observation_sha256(inputs),
            "denoising_calls": 4, "flow_times": [1.0, 0.75, 0.5, 0.25],
            "starts_from_supplied_pure_noise": True, "synthetic_fixture": True,
        }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(timepath, "check_plan", plan)
    monkeypatch.setattr(timepath, "build_jobs", jobs)
    monkeypatch.setattr(evidence, "validate_training_provenance", provenance)
    monkeypatch.setattr(evidence, "_probe_data", probe_data)
    monkeypatch.setattr(evidence, "_load_policy_and_processor", load_model)
    monkeypatch.setattr(sampling, "_make_batches", batches)
    monkeypatch.setattr(sampling, "_sample_counterfactual", sample)
    return args, fixture


@pytest.mark.parametrize(("outcome", "explicit_checkpoint"), [("zero", True), ("adverse", False)])
def test_complete_run_writes_paired_arrays_metrics_and_real_figures(
    tmp_path, monkeypatch, outcome, explicit_checkpoint,
):
    args, fixture = _fixture_run(
        monkeypatch, tmp_path, outcome=outcome, explicit_checkpoint=explicit_checkpoint,
    )
    result = sampling.run(args)
    report = json.loads((args.output / "full_sampling_evidence.json").read_text())
    assert report == result
    assert report["status"] == "completed"
    assert report["protocol"] == sampling.PROTOCOL
    assert report["training_provenance"]["fixture"] is True
    assert fixture.loads == [fixture.checkpoint.resolve()]
    assert fixture.jobs == ([] if explicit_checkpoint else [("time_input", 2)])
    count = len(fixture.anchors) * len(args.noise_seeds)
    assert report["number_of_conditions"] == count == 6
    assert report["number_of_anchors"] == 3 and report["number_of_demonstrations"] == 2
    assert report["episode_ids"] == [9, 74]
    assert report["sampling"]["native_num_inference_steps"] == 4
    assert report["sampling"]["checkpoint_n_action_steps"] == 50
    assert report["sampling"]["executed_prefix_steps"] == 10
    assert report["sampling"]["targets_passed_to_inference"] is False
    assert report["sampling"]["one_checkpoint_loaded_for_all_interventions"] is True
    assert report["regions"] == {"executed_prefix": [0, 10], "full_chunk": [0, 50]}
    assert report["representative_condition"]["global_frame_index"] == 100
    assert report["representative_condition"]["noise_seed"] == 3000

    with np.load(args.output / "full_sampling_arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "clean_target", "initial_noise", "correct_actions", "zero_actions", "reversed_actions",
        }
        assert arrays["clean_target"].shape == (count, 50, 7)
        assert arrays["initial_noise"].shape == (count, 50, 32)
        expected_conditions = [(episode, frame, seed) for episode, frame in fixture.anchors
                               for seed in args.noise_seeds]
        actual_conditions = [(row["episode_index"], row["global_frame_index"], row["noise_seed"])
                             for row in report["conditions"]]
        assert actual_conditions == expected_conditions
        assert len(fixture.calls) == count * 3
        for index, (_, frame, seed) in enumerate(expected_conditions):
            np.testing.assert_array_equal(arrays["clean_target"][index], _synthetic_target(frame))
            expected_noise = sampling.stable_sampling_noise(seed, frame, (1, 50, 32))
            np.testing.assert_array_equal(arrays["initial_noise"][index], expected_noise[0].numpy())
            calls = fixture.calls[index * 3:(index + 1) * 3]
            assert [row["mode"] for row in calls] == list(sampling.MODES)
            for row in calls:
                assert row["frame"] == frame
                torch.testing.assert_close(row["noise"], expected_noise, rtol=0, atol=0)
            audits = report["sampling_audits"][index]
            assert len({value["initial_noise_sha256"] for value in audits.values()}) == 1
            assert len({value["observation_sha256"] for value in audits.values()}) == 1
            assert all(value["denoising_calls"] == 4 for value in audits.values())

        episode_rows = np.asarray([row["episode_index"] for row in report["conditions"]])
        for region, end in (("executed_prefix", 10), ("full_chunk", 50)):
            assert set(report["metrics"][region]) == {"all", "arm", "gripper"}
            for component, dims in (("all", range(7)), ("arm", range(6)), ("gripper", [6])):
                metric = report["metrics"][region][component]
                for mode in sampling.MODES:
                    errors = (arrays[f"{mode}_actions"].astype(np.float64)
                              - arrays["clean_target"].astype(np.float64)) ** 2
                    per_row = errors[:, :end, :][:, :, dims].mean(axis=(1, 2))
                    # Equal-weight demos, despite demo 9 having twice as many windows.
                    expected = [float(per_row[episode_rows == demo].mean()) for demo in (9, 74)]
                    assert metric["per_demo_mse"][mode] == pytest.approx(expected)
                    assert metric["mean_mse"][mode] == pytest.approx(np.mean(expected))
                deltas = metric["mean_paired_deltas"]
                if outcome == "zero" or component == "gripper":
                    assert deltas == {"zero_minus_correct": 0.0, "reversed_minus_correct": 0.0}
                else:
                    assert deltas["zero_minus_correct"] < 0
                    assert deltas["reversed_minus_correct"] < 0

    png = args.output / "fig05_full_sampling_time_alignment.png"
    svg = args.output / "fig05_full_sampling_time_alignment.svg"
    assert report["figure_paths"] == [str(png), str(svg)]
    with Image.open(png) as figure:
        assert figure.format == "PNG"
        assert figure.width >= 1600 and figure.height >= 1000
        assert np.asarray(figure.convert("RGB")).std() > 20
    assert ElementTree.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg"


def test_sampling_failure_replaces_stale_completed_report(tmp_path, monkeypatch):
    args, fixture = _fixture_run(monkeypatch, tmp_path)
    args.output.mkdir(parents=True)
    report_path = args.output / "full_sampling_evidence.json"
    report_path.write_text(json.dumps({"status": "completed", "figure_paths": ["stale.png"]}))
    original = sampling._sample_counterfactual

    def fail_second_arm(policy, inputs, noise, mode):
        if mode == "zero":
            raise RuntimeError("synthetic sampler failure")
        return original(policy, inputs, noise, mode)

    monkeypatch.setattr(sampling, "_sample_counterfactual", fail_second_arm)
    with pytest.raises(RuntimeError, match="synthetic sampler failure"):
        sampling.run(args)
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: synthetic sampler failure"
    assert "figure_paths" not in report and "metrics" not in report
    assert not (args.output / "full_sampling_arrays.npz").exists()
    assert len(fixture.loads) == 1


def test_run_rejects_unpaired_intervention_audits_before_publishing_arrays(tmp_path, monkeypatch):
    args, _ = _fixture_run(monkeypatch, tmp_path)
    original = sampling._sample_counterfactual

    def mismatched_input(policy, inputs, noise, mode):
        value, audit = original(policy, inputs, noise, mode)
        if mode == "reversed":
            audit["initial_noise_sha256"] = "different-synthetic-input"
        return value, audit

    monkeypatch.setattr(sampling, "_sample_counterfactual", mismatched_input)
    with pytest.raises(RuntimeError, match="different paired inputs"):
        sampling.run(args)
    report = json.loads((args.output / "full_sampling_evidence.json").read_text())
    assert report["status"] == "failed"
    assert not (args.output / "full_sampling_arrays.npz").exists()


def write_synthetic_visual_qa(root):
    """Optional developer-only rendered fixture, unmistakably labelled synthetic."""
    from matplotlib.figure import Figure

    root = Path(root)
    with pytest.MonkeyPatch.context() as patch:
        args, _ = _fixture_run(patch, root, outcome="mixed", visual=True)
        args.output = root
        original = Figure.savefig

        def labelled_save(figure, *positional, **kwargs):
            figure.suptitle("SYNTHETIC TEST FIXTURE - layout validation only\n"
                           "Does correct time help full action generation from pure noise?",
                           fontsize=17, weight="bold")
            return original(figure, *positional, **kwargs)

        patch.setattr(Figure, "savefig", labelled_save)
        result = sampling.run(args)
        (root / "SYNTHETIC_FIXTURE_ONLY.txt").write_text(
            "Synthetic artifact test: mock checkpoint, data, model and sampling. "
            "These plots validate layout and reporting only; they are not experiment results.\n",
            encoding="utf-8",
        )
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render explicitly synthetic full-sampling QA fixture")
    parser.add_argument("--visual-qa-dir", type=Path, required=True)
    options = parser.parse_args()
    write_synthetic_visual_qa(options.visual_qa_dir)
