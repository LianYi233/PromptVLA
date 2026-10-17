"""Opt-in training diagnostics for PI05 on absolute-joint Piper datasets."""

import math
from dataclasses import dataclass


@dataclass
class PiperEvalConfig:
    # Full observation-conditioned denoising at fixed recorded observations; no robot execution.
    freq: int = 0
    samples: int = 16  # Per split, uniformly spaced over its dataset rows.
    seed: int = 0
    execution_steps: int = 8
    moving_threshold_deg: float = 1.0
    # Sample the actual optimizer update every N steps (0 disables this instrumentation).
    update_freq: int = 0

    @property
    def enabled(self) -> bool:
        return self.freq > 0 or self.update_freq > 0

    def validate(self) -> None:
        for name in ("freq", "update_freq", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"piper_eval.{name} must be a non-negative integer")
        for name in ("samples", "execution_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"piper_eval.{name} must be a positive integer")
        if not math.isfinite(self.moving_threshold_deg) or self.moving_threshold_deg <= 0:
            raise ValueError("piper_eval.moving_threshold_deg must be finite and positive")
