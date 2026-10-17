"""Known linear residuals, protocol matching and CPU report/dispatch integration."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts.plot_pi05_prompt_rank import make_report
from lerobot.utils.prompt_rank import tangent_spectrum
from lerobot.utils.prompt_rank_analysis import (
    SUITES,
    correlations,
    read_success,
    residual_alignment,
    resolve_runs,
)

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "examples/analysis/analyze_pi05_vlm_rank_sweep.sh"


def test_known_residual_alignment_and_independent_projection_check():
    g = np.diag([3.0, 1.0, 0.0])
    h = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    delta = np.array([1.0, 1.0, 0.0])
    b, c = g @ delta, h @ delta
    result = residual_alignment(g @ g.T, b, h @ g.T, c, 1e-10)
    assert result["top1_alignment"] == pytest.approx(0.9)
    assert result["top2_alignment"] == pytest.approx(1.0)
    assert result["top4_alignment"] is None
    assert result["top_k"][0]["isotropic_reference"] == pytest.approx(1 / 3)
    assert result["heldout_linear_reduction"] == pytest.approx(1.0)
    assert result["linear_delta_norm"] == pytest.approx(np.linalg.norm(delta))
    bad = residual_alignment(g @ g.T, b, h @ g.T, -c, 1e-10)
    assert bad["heldout_linear_reduction"] < -2.9
    orthogonal = residual_alignment(g @ g.T, np.array([0, 0, 1]), h @ g.T, c)
    assert orthogonal["full_projected_range_fraction"] == 0
    assert orthogonal["train_linear_reduction"] == 0
    zero = residual_alignment(np.zeros((3, 3)), b, np.zeros((2, 3)), c)
    assert zero["numerical_rank"] == 0
    assert zero["full_projected_range_fraction"] == 0
    exact = residual_alignment(g @ g.T, np.zeros(3), h @ g.T, np.zeros(2))
    assert exact["heldout_linear_reduction"] is None
    with pytest.raises(ValueError):
        residual_alignment(g @ g.T, b, h @ g.T, c, 0)


def make_inputs(tmp_path):
    counts = [1, 2, 4]
    for count in counts:
        run = tmp_path / "checkpoints" / f"pi05-base-vlm{count}-seed1"
        directory = run / "prompt_rank"
        directory.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "quantity": "test velocity Jacobian",
            "projection": "shared",
            "forward_precision": "float32",
            "parameter_leaf_precision": "float32",
            "torch_version": "test",
            "sample_sha256": "fixed",
            "pretrained_path": "/models/base",
            "normalization_mapping": {},
            "chunk_size": 4,
            "action_dim": 3,
            "projected_rows": 6,
            "scope": "fixed observations",
            "num_vlm_prompt_tokens": count,
            "options": {"samples": 2, "output_probes": 3, "timesteps": [0.5], "seed": 2026},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        records = []
        alignment = run / "prompt_alignment"
        alignment.mkdir()
        protocol = {
            "schema_version": 1,
            "torch_version": "test",
            "device_type": "cpu",
            "heldout_probes": 3,
            "heldout_seed": 2027,
            "ridge_relative": 0.01,
            "gram_rtol": 0.02,
            "scope": "projection",
            "noise_projection_sha256": "fixed",
            "run": run.name,
            "probe_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        }
        (alignment / "protocol.json").write_text(json.dumps(protocol))
        for step in (0, 1500, 3000):
            g = np.zeros((6, count * 8))
            g[:, :6] = np.diag(np.linspace(1, count + step / 3000 + 1, 6))
            metrics, gram, eigenvalues, singular_values = tangent_spectrum(g)
            records.append({"step": step, **metrics})
            source = directory / f"step_{step:06d}.npz"
            np.savez(
                source,
                gram=gram,
                eigenvalues=eigenvalues,
                singular_values=singular_values,
                prompt=np.ones((count, 8)),
            )
            scores = residual_alignment(gram, np.arange(6), gram, np.arange(6))
            scores.update(
                step=step,
                tokens=count,
                seed=1,
                run=run.name,
                snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            (alignment / f"step_{step:06d}.json").write_text(json.dumps(scores))
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
        evaluation = tmp_path / "eval" / f"libero060-all-{run.name}-003000-libero-all4-resume"
        evaluation.mkdir(parents=True)
        tasks = [
            {
                "task_group": suite,
                "task_id": task,
                "metrics": {"successes": [True] * (5 + count) + [False] * (5 - count)},
            }
            for suite in SUITES
            for task in range(10)
        ]
        (evaluation / "eval_info.json").write_text(json.dumps({"per_task": tasks}))
        args = {
            "seed": 1000,
            "eval.n_episodes": 10,
            "env.task": ",".join(SUITES),
            "policy.n_action_steps": 10,
            "env.control_mode": "relative",
            "policy.path": str(run / "checkpoints/003000/pretrained_model"),
        }
        (evaluation / "resume_manifest.json").write_text(
            json.dumps(
                {
                    "arguments": [f"--{key}={value}" for key, value in args.items()],
                    "evaluator_sha256": "fixed-evaluator",
                    "model_files": [],
                }
            )
        )
    return SimpleNamespace(
        ckpt_root=tmp_path / "checkpoints",
        tokens=counts,
        seed=1,
        run_prefix="pi05",
        source="base",
        final_step=3000,
        steps=[0, 1500, 3000],
        eval_root=tmp_path / "eval",
        eval_seed=1000,
        episodes_per_task=10,
        include_alignment=True,
        analysis_name="prompt_alignment",
        font_path=None,
        output_dir=tmp_path / "report",
    )


def test_full_cpu_report_pairs_success_and_alignment_and_writes_figures(tmp_path):
    args = make_inputs(tmp_path)
    make_report(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert [row["overall_success_pct"] for row in report["final"]] == [60, 70, 90]
    assert all(row["overall_episodes"] == 400 for row in report["final"])
    for filename in ("rank_dynamics", "spectra_and_energy", "residual_alignment", "success_vs_geometry"):
        for extension in ("png", "pdf"):
            assert (args.output_dir / f"{filename}.{extension}").stat().st_size > 1000
    assert len(report["correlations"]) == 25  # This toy probe has <16 directions.


def test_mismatched_protocol_and_incomplete_eval_are_rejected(tmp_path):
    args = make_inputs(tmp_path)
    runs = resolve_runs(args.ckpt_root, args.tokens, args.seed)
    evaluation = args.eval_root / f"libero060-all-{runs[0]['name']}-003000-libero-all4-resume/eval_info.json"
    document = json.loads(evaluation.read_text())
    document["per_task"].pop()
    evaluation.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Missing tasks"):
        read_success(args.eval_root, runs[0], 3000, 10, 1000)
    path = runs[1]["path"] / "prompt_rank/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sample_sha256"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unpaired"):
        resolve_runs(args.ckpt_root, args.tokens, args.seed)


def test_tied_spearman_and_degenerate_correlations():
    rows = [{"x": x, "overall_success_pct": y} for x, y in [(1, 2), (1, 2), (2, 1), (3, 0)]]
    assert correlations(rows, ["x"])[0]["spearman"] == pytest.approx(-1)
    for row in rows:
        row["x"] = 1
    assert correlations(rows, ["x"])[0]["pearson"] is None


def test_wrapper_dispatches_seed1_six_tokens_and_all_stages_without_mutation(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "CKPT_ROOT": str(tmp_path / "checkpoints"),
        "BASE_OUTPUT": str(tmp_path / "eval"),
        "REPORT_ROOT": str(tmp_path / "report"),
        "GPU_ID": "1",
    }
    for key in ("VLM_PROMPT_TOKEN_COUNTS", "SOURCE", "RUN_PREFIX", "ALIGNMENT_STEPS"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(WRAPPER), "all", "1"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert "run_eval_libero_vlm_only_token_sweep.sh base all 1" in lines[0]
    assert "CUDA_VISIBLE_DEVICES=1" in lines[1]
    assert "--tokens 1 2 4 8 16 32" in lines[1]
    assert "--steps 0 1500 3000" in lines[1]
    assert "--include-alignment" in lines[2]
    assert list(tmp_path.iterdir()) == []
