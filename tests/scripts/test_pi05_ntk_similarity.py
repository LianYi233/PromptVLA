"""CPU-only checks for the CKA estimator, paired checkpoints and heatmap exports."""

import copy
import json

import numpy as np
import pytest

from lerobot.scripts.plot_pi05_ntk_similarity import centered_unit_kernel, compute_similarity, plot_similarity


def _kernel(angle):
    first = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2)
    second = np.asarray([1.0, 1.0, -2.0]) / np.sqrt(6)
    direction = np.cos(angle) * first + np.sin(angle) * second
    return np.outer(direction, direction)


def _document(stage_count=4, scopes=("backbone", "prompts")):
    stages = [
        {"name": name, "total_step": step}
        for name, step in zip(("before", "priming", "stage2", "final"), (0, 750, 1000, 4000), strict=True)
    ][:stage_count]
    records = []
    for index, stage in enumerate(stages):
        for seed in (0, 1):
            groups = {}
            for scope in scopes:
                for module, rate in (("vlm", 0.1), ("action", 0.5)):
                    # Independent stage scales must not affect CKA.
                    kernel = _kernel(index * rate * (seed + 1)) * (index + 1) * 10 ** (seed * 3)
                    groups[f"{scope}/{module}"] = {
                        "kernel": kernel.tolist(),
                        "parameter_representation": "exact" if scope == "prompts" else "countsketch",
                        "parameter_count": 5,
                        "effective_rank": 1.0,
                        "participation_rank": 1.0,
                        "parameter_normalized_energy": float(np.trace(kernel) / 5),
                        "tangent_trace": float(np.trace(kernel)),
                        "sketch_trace_relative_error": 0.0,
                    }
            records.append(
                {
                    **stage,
                    "local_step": 0,
                    "seed": seed,
                    "groups": groups,
                    "stage": stage["name"],
                    "parameter_groups": {group: [["weight", [5]]] for group in groups},
                }
            )
    return {
        "manifest": {
            "stages": stages,
            "seeds": [0, 1],
            "sample_indices": [10, 20, 30],
            "sample_digest": "synthetic-test-only",
            "protocol": "shared-Rademacher output NTK; synthetic test fixture",
        },
        "records": records,
    }


def test_cka_known_geometry_centering_and_scale_invariance():
    first, second = _kernel(0), _kernel(0.7)
    a, b = centered_unit_kernel(first), centered_unit_kernel(second)
    assert np.sum(a * b) == pytest.approx(np.cos(0.7) ** 2)
    # Same rank and eigenvalues do not imply the same tangent geometry.
    np.testing.assert_allclose(np.linalg.eigvalsh(first), np.linalg.eigvalsh(second), atol=1e-15)
    assert np.sum(a * b) < 0.7
    np.testing.assert_allclose(centered_unit_kernel(first * 1e-250), a, atol=1e-14)
    np.testing.assert_allclose(centered_unit_kernel(first * 1e250), a, atol=1e-14)
    np.testing.assert_allclose(centered_unit_kernel(first + 100 * np.ones((3, 3))), a, atol=1e-12)
    assert np.sum(a * centered_unit_kernel(_kernel(np.pi / 2))) == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize(
    "kernel,message",
    [
        (np.zeros((3, 3)), "zero kernel"),
        (np.ones((3, 3)), "constant"),
        (np.eye(3) * np.nan, "non-finite"),
        (np.ones((2, 3)), "square"),
        (np.eye(1), "at least two"),
        (np.asarray([[1, 0.5], [0.1, 1]]), "symmetric"),
        (np.diag([1, -1, 1]), "positive semidefinite"),
    ],
)
def test_invalid_or_degenerate_grams_are_not_shown_as_perfect_alignment(kernel, message):
    with pytest.raises(ValueError, match=message):
        centered_unit_kernel(kernel)


def test_seed_pairing_and_aggregation_do_not_depend_on_record_order():
    document = _document()
    document["records"].reverse()
    summary, pairs = compute_similarity(document)
    expected = np.cos([0.5, 1.0]) ** 2
    group = summary["groups"]["backbone/action"]
    assert group["median"][0][1] == pytest.approx(np.median(expected))
    assert group["q25"][0][1] == pytest.approx(np.quantile(expected, 0.25))
    assert group["q75"][0][1] == pytest.approx(np.quantile(expected, 0.75))
    assert group["per_seed"][0][0][1] == pytest.approx(expected[0])
    assert len(pairs) == 4 * 2 * 6
    for metrics in summary["groups"].values():
        matrix = np.asarray(metrics["median"])
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1)
    # Averaging kernels first is a different estimator.
    averaged = [np.mean([_kernel(index * 0.5 * (seed + 1)) for seed in (0, 1)], axis=0) for index in range(2)]
    wrong = np.sum(centered_unit_kernel(averaged[0]) * centered_unit_kernel(averaged[1]))
    assert not np.isclose(group["median"][0][1], wrong)


@pytest.mark.parametrize("change", ["missing", "duplicate", "step", "shape", "signature", "kernel", "order"])
def test_incompatible_records_fail_instead_of_mixing_experiments(change):
    document = _document()
    first = document["records"][0]
    if change == "missing":
        document["records"].pop()
    elif change == "duplicate":
        document["records"].append(copy.deepcopy(first))
    elif change == "step":
        first["total_step"] = 123
    elif change == "shape":
        first["groups"]["backbone/vlm"]["kernel"] = np.eye(4).tolist()
    elif change == "signature":
        first["parameter_groups"]["backbone/vlm"] = [["other_weight", [5]]]
    elif change == "kernel":
        del first["groups"]["backbone/vlm"]["kernel"]
    else:
        del document["manifest"]["sample_indices"]
    with pytest.raises(ValueError):
        compute_similarity(document)


def test_auto_scope_and_partial_stages():
    summary, pairs = compute_similarity(_document(3, ("prompts",)))
    assert set(summary["groups"]) == {"prompts/vlm", "prompts/action"}
    assert np.asarray(summary["groups"]["prompts/vlm"]["median"]).shape == (3, 3)
    assert len(pairs) == 2 * 2 * 3
    with pytest.raises(ValueError, match="Missing saved kernel"):
        compute_similarity(_document(3, ("prompts",)), "both")
    with pytest.raises(ValueError, match="two to four"):
        compute_similarity(_document(1))


@pytest.mark.parametrize("stage_count,full_matrix", [(3, False), (4, True)])
def test_exports_with_no_model_or_loss_files(tmp_path, stage_count, full_matrix):
    pytest.importorskip("matplotlib")
    source = tmp_path / "results.json"
    source.write_text(json.dumps(_document(stage_count)))
    original = source.read_bytes()
    summary = plot_similarity(source, scope="backbone", full_matrix=full_matrix)
    assert source.read_bytes() == original
    out = tmp_path / "similarity"
    for name in (
        "backbone_stage_similarity",
        "backbone_vlm_stage_similarity",
        "backbone_action_stage_similarity",
    ):
        for extension in ("png", "pdf", "svg"):
            assert (out / f"{name}.{extension}").stat().st_size > 1000
    saved = json.loads((out / "stage_similarity.json").read_text())
    assert saved == summary
    assert saved["plot"]["color_limits"] == [0, 1]
    expected_pairs = 2 * 2 * stage_count * (stage_count - 1) // 2
    assert len((out / "stage_similarity_pairs.csv").read_text().splitlines()) == 1 + expected_pairs


def test_existing_plot_entry_point_also_exports_similarity(tmp_path):
    pytest.importorskip("matplotlib")
    from lerobot.scripts.plot_pi05_ntk_stages import plot_results

    source = tmp_path / "results.json"
    source.write_text(json.dumps(_document(3, ("backbone",))))
    plot_results(source)
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert (tmp_path / "similarity/backbone_stage_similarity.png").exists()
