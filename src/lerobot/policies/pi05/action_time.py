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

    def __init__(
        self, width: int, encoding_dim: int = 32, fps: float = 20.0, *,
        min_frequency_hz: float = 0.1, max_frequency_hz: float = 10.0,
        gain_mode: str = "none",
    ):
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
        if (
            not math.isfinite(min_frequency_hz)
            or not math.isfinite(max_frequency_hz)
            or min_frequency_hz <= 0
            or max_frequency_hz <= min_frequency_hz
        ):
            raise ValueError("Time frequencies must satisfy 0 < min_frequency_hz < max_frequency_hz")
        if gain_mode not in ("none", "relative_rms"):
            raise ValueError("gain_mode must be none or relative_rms")
        self.encoding_dim = encoding_dim
        self.fps = float(fps)
        self.min_frequency_hz = float(min_frequency_hz)
        self.max_frequency_hz = float(max_frequency_hz)
        self.gain_mode = gain_mode
        # Still exactly width*encoding_dim learnable parameters, no RNG draw.
        self.weight = nn.Parameter(torch.zeros(width, encoding_dim))

    def features(
        self,
        horizon: int,
        device: torch.device,
        *,
        delta_t: Tensor | None = None,
        basis: str = "index",
    ) -> Tensor:
        """Index features [H,32] or exact per-sample temporal features [B,H,32].

        index: φ(j/fps), independent of true sample cadence;
        global_dt: φ(Δt) at every token, a per-sample constant;
        physical: φ(j·Δt) at each target timestamp.
        All share the SAME 32 features and zero-initialized projection.
        """
        if basis not in ("index", "global_dt", "physical"):
            raise ValueError("Unknown action time basis")
        offsets = torch.arange(horizon, dtype=torch.float32, device=device)
        if basis == "index":
            seconds = offsets / self.fps
        else:
            if delta_t is None:
                raise ValueError("Per-sample physical/global dt requires pi05_action_dt in batch")
            delta = torch.as_tensor(delta_t, dtype=torch.float32, device=device)
            if delta.ndim != 1 or not torch.isfinite(delta).all() or not (delta > 0).all():
                raise ValueError("delta_t must be finite positive tensor [batch]")
            seconds = (
                delta[:, None].expand(-1, horizon)
                if basis == "global_dt"
                else delta[:, None] * offsets[None, :]
            )
        frequencies = torch.logspace(
            math.log10(self.min_frequency_hz),
            math.log10(self.max_frequency_hz),
            self.encoding_dim // 2,
            device=device,
        )
        phases = 2 * math.pi * seconds[..., None] * frequencies
        return torch.cat((phases.sin(), phases.cos()), dim=-1)

    def forward(
        self, action_embeddings: Tensor, *,
        delta_t: Tensor | None = None, basis: str = "index",
    ) -> Tensor:
        if action_embeddings.ndim != 3 or action_embeddings.shape[-1] != self.weight.shape[0]:
            raise ValueError("action_embeddings must have shape (batch, horizon, expert_width)")
        features = self.features(
            action_embeddings.shape[1], action_embeddings.device,
            delta_t=delta_t, basis=basis,
        )
        if features.ndim == 3 and features.shape[0] != action_embeddings.shape[0]:
            raise ValueError("delta_t length differs from action batch size")
        correction = F.linear(features.to(dtype=self.weight.dtype), self.weight)
        if correction.ndim == 2:
            correction = correction.unsqueeze(0)
        if self.gain_mode == "relative_rms":
            # A bounded, per-token content-dependent gain. Normalization
            # over the chunk keeps its typical amplitude near one; detach
            # prevents the gate from becoming a second gradient path
            # through the noisy-action input. No learnable gate parameters.
            with torch.no_grad():
                token_rms = action_embeddings.detach().float().square().mean(
                    dim=-1, keepdim=True
                ).add(1e-6).sqrt()
                chunk_scale = token_rms.mean(dim=1, keepdim=True).clamp_min(1e-4)
                gains = (token_rms / chunk_scale).clamp(0.5, 2.0)
            correction = correction * gains.to(dtype=correction.dtype)
        return action_embeddings + correction.to(dtype=action_embeddings.dtype)



def make_temporal_projection_hook(
    encoder: RelativeActionTimeEncoding,
    *,
    horizon: int,
):
    """Inject W·sinusoidal_time through an EXISTING Q/K/V projection.

    The hook is stateless: it derives the residual entirely from output length,
    encoder.weight and the hooked projection's own weight. In particular it
    remains correct when gradient-checkpointing recomputes a layer at backward.
    This modifies neither projection parameters nor state_dict key names.
    It must be installed on a FIRST-LAYER Action Expert projection only, and
    is intentionally restricted to experiments with zero expert prompts.
    """

    def hook(projection: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        if not isinstance(output, Tensor) or output.ndim != 3 or output.shape[1] != horizon:
            raise RuntimeError(
                "Time-QK/V hook expected only the first-layer action tokens "
                f"(batch,{horizon},projection_width); got {getattr(output, 'shape', None)}"
            )
        if not isinstance(projection, nn.Linear):
            raise TypeError("Time-QK/V requires the native nn.Linear expert projection")
        # Identical 32->1024 learnable matrix and fixed temporal basis as
        # Time-Input. The native pretrained Q/K/V matrix is REUSED, not copied.
        features = encoder.features(horizon, output.device)
        residual = F.linear(features.to(dtype=encoder.weight.dtype), encoder.weight)
        projected = F.linear(residual.to(dtype=projection.weight.dtype), projection.weight)
        return output + projected.unsqueeze(0).to(dtype=output.dtype)

    return hook
