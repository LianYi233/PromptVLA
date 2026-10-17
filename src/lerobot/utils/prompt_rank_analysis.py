"""CPU-only validation and residual-alignment math for fixed prompt-rank probes."""

import csv
import json
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MATCH_FIELDS = (
    "schema_version",
    "quantity",
    "projection",
    "forward_precision",
    "parameter_leaf_precision",
    "torch_version",
    "sample_sha256",
    "pretrained_path",
    "normalization_mapping",
    "chunk_size",
    "action_dim",
    "projected_rows",
    "scope",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def resolve_runs(root, tokens, seed, prefix="pi05", source="base"):
    if seed < 0 or not tokens or len(set(tokens)) != len(tokens) or min(tokens) < 1:
        raise ValueError("Expected distinct positive token counts and a nonnegative seed")
    runs = []
    for count in tokens:
        path = Path(root) / f"{prefix}-{source}-vlm{count}-seed{seed}"
        manifest = read_json(path / "prompt_rank/manifest.json")
        if manifest["num_vlm_prompt_tokens"] != count:
            raise ValueError(f"Token count mismatch: {path}")
        runs.append({"path": path, "name": path.name, "tokens": count, "seed": seed, "manifest": manifest})
    reference = runs[0]["manifest"]
    for run in runs[1:]:
        current = run["manifest"]
        mismatches = [key for key in MATCH_FIELDS if current[key] != reference[key]]
        for key in ("samples", "output_probes", "timesteps", "seed"):
            if current["options"][key] != reference["options"][key]:
                mismatches.append(f"options.{key}")
        if mismatches:
            raise ValueError(f"Unpaired probe protocol in {run['name']}: {mismatches}")
    return runs


def snapshot(run, step):
    path = run["path"] / f"prompt_rank/step_{step:06d}.npz"
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("gram", "eigenvalues", "singular_values", "prompt")}
    n = run["manifest"]["projected_rows"]
    if result["gram"].shape != (n, n) or result["eigenvalues"].shape != (n,):
        raise ValueError(f"Snapshot dimensions differ from protocol: {path}")
    if result["prompt"].shape[0] != run["tokens"] or any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError(f"Invalid prompt/spectrum: {path}")
    return result


def metric_history(run):
    rows = [
        json.loads(line)
        for line in (run["path"] / "prompt_rank/metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    steps = [row["step"] for row in rows]
    if not rows or len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError(f"Empty, duplicate or unordered history: {run['name']}")
    for row in rows:
        if row["row_count"] != run["manifest"]["projected_rows"] or any(
            not np.isfinite(value) for value in row.values()
        ):
            raise ValueError(f"Invalid metrics: {run['name']}")
    return rows


def read_success(eval_root, run, step, episodes, eval_seed):
    directory = Path(eval_root) / f"libero060-all-{run['name']}-{step:06d}-libero-all4-resume"
    document = read_json(directory / "eval_info.json")
    manifest = read_json(directory / "resume_manifest.json")
    args = {
        arg[2:].split("=", 1)[0]: arg.split("=", 1)[1]
        for arg in manifest["arguments"]
        if arg.startswith("--") and "=" in arg
    }
    expected = {
        "seed": str(eval_seed),
        "eval.n_episodes": str(episodes),
        "env.task": ",".join(SUITES),
        "policy.n_action_steps": "10",
        "env.control_mode": "relative",
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Evaluation protocol differs: {directory}")
    checkpoint = run["path"] / f"checkpoints/{step:06d}/pretrained_model"
    if Path(args.get("policy.path", "")).resolve() != checkpoint.resolve():
        raise ValueError(f"Evaluation points to a different checkpoint: {directory}")
    if not manifest.get("evaluator_sha256"):
        raise ValueError(f"Missing evaluator provenance: {directory}")
    if checkpoint.is_dir():
        current_files = [
            [str(p.relative_to(checkpoint)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file()
        ]
        if current_files != manifest["model_files"]:
            raise ValueError(f"Checkpoint files changed after evaluation: {directory}")
    tasks = document.get("per_task") or document.get("Aggregated Metrics for per_task")
    if not isinstance(tasks, list):
        raise ValueError(f"Missing task results: {directory}")
    expected_tasks = {(suite, task) for suite in SUITES for task in range(10)}
    seen, values = set(), {suite: [] for suite in SUITES}
    for task in tasks:
        key = (task["task_group"], task["task_id"])
        successes = task.get("metrics", task).get("successes")
        if (
            key not in expected_tasks
            or key in seen
            or not isinstance(successes, list)
            or (len(successes) != episodes or any(type(x) is not bool for x in successes))
        ):
            raise ValueError(f"Incomplete/duplicate/invalid task results: {key}")
        seen.add(key)
        values[key[0]].extend(successes)
    if seen != expected_tasks:
        raise ValueError(f"Missing tasks: {expected_tasks - seen}")
    values["overall"] = [value for suite in SUITES for value in values[suite]]
    summary = {}
    for suite, outcomes in values.items():
        summary.update(
            {
                f"{suite}_success_pct": 100 * sum(outcomes) / len(outcomes),
                f"{suite}_successes": sum(outcomes),
                f"{suite}_episodes": len(outcomes),
            }
        )
    return summary, manifest["evaluator_sha256"]


def residual_alignment(gram, residual, cross, heldout_residual, ridge_relative=0.01):
    """Top-k residual overlap and a ridge linear correction checked on independent output probes.

    G and H share observations/noise but use independent output projection vectors.
    gram=G G^T, cross=H G^T, b=S(target-v), c=T(target-v).
    No policy weights are updated. Held-out scores are not held-out-task performance.
    """
    gram, b, cross, c = [np.asarray(x, dtype=np.float64) for x in (gram, residual, cross, heldout_residual)]
    if (
        gram.shape != (b.size, b.size)
        or b.ndim != 1
        or c.ndim != 1
        or cross.shape != (c.size, b.size)
        or b.size == 0
        or c.size == 0
        or not all(np.isfinite(x).all() for x in (gram, b, cross, c))
        or not np.isfinite(ridge_relative)
        or ridge_relative <= 0
    ):
        raise ValueError("Invalid alignment arrays or ridge strength")
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    cutoff = np.finfo(np.float64).eps * len(b) * max(eigenvalues[0], 0)
    if eigenvalues.min() < -max(cutoff * 100, 1e-30):
        raise ValueError("Alignment Gram matrix is not positive semidefinite")
    positive = eigenvalues > cutoff
    lam, u = eigenvalues[positive], vectors[:, positive]
    b2, c2 = float(b @ b), float(c @ c)
    coefficients = u.T @ b
    cumulative = np.cumsum(coefficients**2) / b2 if b2 > 0 else np.zeros(len(lam))
    top = [
        {
            "k": k,
            "captured_fraction": float(np.clip(cumulative[k - 1], 0, 1)) if b2 else None,
            "isotropic_reference": k / len(b),
        }
        for k in range(1, len(lam) + 1)
    ]
    trace = float(lam.sum())
    ridge = ridge_relative * trace / len(b)
    alpha = u @ (coefficients / (lam + ridge)) if trace else np.zeros_like(b)
    train_error, heldout_error = b - gram @ alpha, c - cross @ alpha
    result = {
        "numerical_rank": len(lam),
        "row_count": len(b),
        "ridge_relative": ridge_relative,
        "ridge_absolute": ridge,
        "projected_residual_energy": b2,
        "heldout_projected_residual_energy": c2,
        "train_linear_reduction": float(1 - (train_error @ train_error) / b2) if b2 else None,
        "heldout_linear_reduction": float(1 - (heldout_error @ heldout_error) / c2) if c2 else None,
        "linear_delta_norm": float(np.sqrt(max(alpha @ gram @ alpha, 0))),
        "full_projected_range_fraction": (float(np.clip(cumulative[-1], 0, 1)) if len(lam) else 0.0)
        if b2
        else None,
        "top_k": top,
    }
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        result[f"top{k}_alignment"] = top[k - 1]["captured_fraction"] if k <= len(top) else None
    for threshold in (90, 95):
        k = int(np.searchsorted(np.cumsum(lam) / trace, threshold / 100) + 1) if trace else 0
        result[f"rank{threshold}_energy"] = k
        result[f"alignment_at_rank{threshold}"] = top[k - 1]["captured_fraction"] if k else None
    return result


def correlations(rows, metrics):
    """Descriptive correlations only; six token settings do not establish causality."""

    def ranks(values):
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        del unique
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    def coefficient(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) and np.ptp(y) else None

    result = []
    for metric in metrics:
        for suite in (*SUITES, "overall"):
            selected = [
                row
                for row in rows
                if row.get(metric) is not None and row.get(f"{suite}_success_pct") is not None
            ]
            if len(selected) < 3:
                continue
            x = np.array([row[metric] for row in selected])
            y = np.array([row[f"{suite}_success_pct"] for row in selected])
            result.append(
                {
                    "metric": metric,
                    "suite": suite,
                    "n": len(x),
                    "pearson": coefficient(x, y),
                    "spearman": coefficient(ranks(x), ranks(y)),
                }
            )
    return result
