"""Physical action offsets, independent of the flow-matching noise timestep.

The experiment keeps every action chunk contiguous. Its offsets are therefore
``0, 1 / fps, ..., (H - 1) / fps`` both in training and at inference. No episode
clock, demonstration labels, or future event times are inputs to this module.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class RelativeActionTimeEncoding(nn.Module):
    """Add a learned projection of fixed sinusoidal relative-time features.

    Zero initialization preserves the source policy at step zero. Constructing
    the parameter directly also consumes no random numbers, keeping all shared
    initial weights and the subsequent random stream identical to time-off runs.
    At fixed fps these features are redundant with token position; their purpose
    is to test an additional inductive bias, not to repair reordered labels.
    """

    def __init__(self, width: int, encoding_dim: int = 32, fps: float = 20.0):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if (
            isinstance(encoding_dim, bool)
            or not isinstance(encoding_dim, int)
            or encoding_dim < 2
            or encoding_dim % 2
        ):
            raise ValueError("encoding_dim must be a positive even integer of at least 2")
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.encoding_dim = encoding_dim
        self.fps = float(fps)
        self.weight = nn.Parameter(torch.zeros(width, encoding_dim))

    def features(self, horizon: int, device: torch.device) -> Tensor:
        """Return H x D features with fixed frequencies from 0.1 to 10 Hz."""
        offsets = torch.arange(horizon, dtype=torch.float32, device=device) / self.fps
        frequencies = torch.logspace(-1, 1, self.encoding_dim // 2, device=device)
        phases = 2 * math.pi * offsets[:, None] * frequencies[None, :]
        return torch.cat((phases.sin(), phases.cos()), dim=-1)

    def forward(self, action_embeddings: Tensor) -> Tensor:
        if action_embeddings.ndim != 3 or action_embeddings.shape[-1] != self.weight.shape[0]:
            raise ValueError("action_embeddings must have shape (batch, horizon, expert_width)")
        features = self.features(action_embeddings.shape[1], action_embeddings.device)
        correction = F.linear(features.to(dtype=self.weight.dtype), self.weight)
        return action_embeddings + correction[None].to(dtype=action_embeddings.dtype)
