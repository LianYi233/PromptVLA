"""Opt-in fixed-probe tangent diagnostics for PI05 VLM-only prompt training."""

import math
from dataclasses import dataclass, field


@dataclass
class PromptRankConfig:
    # Zero disables monitoring. Enabled runs also measure before training and at the final step.
    freq: int = 0
    samples: int = 16
    output_probes: int = 4
    timesteps: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    # Independent of training seed and token count, for paired comparisons.
    seed: int = 2026
    extra_steps: list[int] = field(default_factory=list)
    fixed_flow_loss: bool = False

    def validate(self) -> None:
        for name in ("freq", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"prompt_rank.{name} must be a non-negative integer")
        for name in ("samples", "output_probes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"prompt_rank.{name} must be a positive integer")
        if not self.timesteps or any(not math.isfinite(t) or not 0 < t < 1 for t in self.timesteps):
            raise ValueError("prompt_rank.timesteps must be a nonempty list strictly between 0 and 1")
        if len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("prompt_rank.timesteps must be distinct")
        if any(type(step) is not int or step < 0 for step in self.extra_steps):
            raise ValueError("prompt_rank.extra_steps must contain nonnegative integers")
