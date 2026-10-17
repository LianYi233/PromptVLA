"""Small, dependency-free log-window diagnostics for offline training analysis."""

import json
import math
from numbers import Real
from pathlib import Path


class TrainingDiagnostics:
    """Aggregate scalar/list policy metrics without another forward or tensor transfer.

    Only the main process owns this writer. Policy metrics therefore describe rank 0;
    the separately supplied MetricsTracker values include its existing DDP reductions.
    Nonfinite observations are counted, never written as non-standard JSON NaN/Infinity.
    """

    def __init__(self, path: Path, context: dict):
        self.path = Path(path)
        self.context = context
        self.values = {}
        self.updates = 0

    def update(self, metrics: dict | None):
        self.updates += 1
        metrics = {"optimizer_step/skipped": 0.0, **(metrics or {})}

        def add(name, value):
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    add(f"{name}/{index}", item)
            elif isinstance(value, Real):
                total, count, low, high, nonfinite = self.values.get(name, (0.0, 0, math.inf, -math.inf, 0))
                value = float(value)
                if math.isfinite(value):
                    total, count = total + value, count + 1
                    low, high = min(low, value), max(high, value)
                else:
                    nonfinite += 1
                self.values[name] = (total, count, low, high, nonfinite)

        for name, value in metrics.items():
            add(name, value)

    def write(self, step: int, train_metrics: dict):
        if not self.updates:
            return
        diagnostics = {
            name: {
                "mean": total / count if count else None,
                "min": low if count else None,
                "max": high if count else None,
                "count": count,
                "nonfinite": nonfinite,
            }
            for name, (total, count, low, high, nonfinite) in self.values.items()
        }
        train = {
            name: float(value) if math.isfinite(float(value)) else None
            for name, value in train_metrics.items()
            if isinstance(value, Real)
        }
        record = {
            **self.context,
            "step": step,
            "window_updates": self.updates,
            "train": train,
            "diagnostics_rank0": diagnostics,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.values.clear()
        self.updates = 0
        return record


def diagnostic_means(record: dict) -> dict[str, float]:
    """Scalar window means for W&B, including flattened per-action-dimension losses."""
    metrics = {"diagnostics/window_updates": record["window_updates"]}
    for name, value in record["diagnostics_rank0"].items():
        # A policy's rank-local loss must not overwrite the trainer's reduced DDP loss.
        key = f"policy_rank0/{name}" if name in record["train"] else name
        if value["mean"] is not None:
            metrics[key] = value["mean"]
        if value["nonfinite"]:
            metrics[f"{key}/nonfinite"] = value["nonfinite"]
    return metrics
