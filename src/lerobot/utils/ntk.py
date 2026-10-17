"""Empirical, output-projected NTK utilities for paired checkpoint comparisons.

No loss gradients or optimizer steps are used. CountSketch compresses only the
parameter axis; tangent energy is accumulated BEFORE this compression.
"""

import hashlib

import numpy as np


def spectral_metrics(kernel, parameter_count, tangent_trace=None):
    """Entropy effective rank and Tr(K)/P; the zero kernel has rank/energy zero."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or not np.isfinite(kernel).all():
        raise ValueError("Expected a finite square Gram matrix")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2)
    if eigenvalues.min(initial=0) < -1e-8 * max(np.abs(eigenvalues).max(initial=0), 1e-30):
        raise ValueError("The NTK Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0)
    trace = float(eigenvalues.sum())
    energy_trace = trace if tangent_trace is None else float(tangent_trace)
    if not np.isfinite(energy_trace) or energy_trace < 0:
        raise ValueError("Non-finite or negative tangent trace")
    if trace > 0:
        probabilities = eigenvalues[eigenvalues > 0] / trace
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        participation_rank = float(1 / np.sum(probabilities**2))
    else:
        effective_rank = participation_rank = 0.0
    return {
        "effective_rank": effective_rank,
        "participation_rank": participation_rank,
        "parameter_normalized_energy": energy_trace / parameter_count,
        "tangent_trace": energy_trace,
        "sketched_kernel_trace": trace,
        "sketch_trace_relative_error": abs(trace - energy_trace) / energy_trace if energy_trace else 0.0,
        "parameter_count": int(parameter_count),
        "eigenvalues": eigenvalues[::-1].tolist(),
    }


def parameter_groups(policy, scope="both"):
    """Disjoint groups; action backbone includes the expert and action/time projections."""
    model = policy.model
    backbone = model.paligemma_with_expert
    modules = {}
    if scope in ("backbone", "both"):
        modules["backbone/vlm"] = [backbone.paligemma]
        modules["backbone/action"] = [
            backbone.gemma_expert,
            model.action_in_proj,
            model.action_out_proj,
            model.time_mlp_in,
            model.time_mlp_out,
        ]
    if scope in ("prompts", "both"):
        modules["prompts/vlm"] = [model.vlm_prompt_tokens]
        modules["prompts/action"] = [model.prompt_tokens]
    names = {id(p): name for name, p in policy.named_parameters()}
    seen = set()
    groups = {}
    for group, members in modules.items():
        params = {}
        for module in members:
            for parameter in module.parameters():
                if parameter.numel():
                    params[id(parameter)] = parameter
        if not params or seen.intersection(params):
            raise ValueError(f"Empty or overlapping NTK group: {group}")
        seen.update(params)
        groups[group] = sorted((names[key], p) for key, p in params.items())
    if not groups:
        raise ValueError(f"Unknown parameter scope: {scope}")
    return groups


def enable_analysis_gradients(policy, groups):
    """Call eval FIRST (PI05.train/eval re-freezes weights), then enable diagnostic gradients."""
    policy.eval()
    policy.model.gradient_checkpointing_disable()
    # Requires-grad alone is insufficient: the observation fast path explicitly detaches KV.
    policy.model.paligemma_with_expert.separate_frozen_observations = False
    policy.requires_grad_(False)
    for members in groups.values():
        for _, parameter in members:
            parameter.requires_grad_(True)


def compress_gradients(named_parameters, gradients, sketch_dim, sketch_seed, chunk_size=262144):
    """Stream each VJP into a CountSketch (0 = exact), also returning its exact squared norm.

    Hashes use the stable parameter name and coordinate, not Python's randomized hash.
    The same map MUST be used for every sample, checkpoint and output probe.
    """
    import torch

    if sketch_dim < 0 or chunk_size < 1:
        raise ValueError("Invalid sketch or chunk size")
    device = named_parameters[0][1].device
    size = sketch_dim or sum(p.numel() for _, p in named_parameters)
    result = torch.zeros(size, dtype=torch.float32, device=device)
    norm = torch.zeros((), dtype=torch.float64, device=device)
    offset = 0
    for (name, parameter), grad in zip(named_parameters, gradients, strict=True):
        if grad is not None:
            flat = grad.detach().reshape(-1)
            salt = int.from_bytes(hashlib.sha256(f"{sketch_seed}:{name}".encode()).digest()[:4], "little")
            for start in range(0, flat.numel(), chunk_size):
                values = flat[start : start + chunk_size].float()
                norm += values.double().square().sum()
                if sketch_dim:
                    index = torch.arange(start, start + values.numel(), dtype=torch.int64, device=device)
                    hashed = (index ^ salt) & 0xFFFFFFFF
                    for _ in range(2):
                        hashed = (((hashed >> 16) ^ hashed) * 0x45D9F3B) & 0xFFFFFFFF
                    hashed = (hashed >> 16) ^ hashed
                    signs = 1.0 - 2.0 * ((hashed >> 31) & 1).float()
                    buckets = (hashed & 0x7FFFFFFF) % sketch_dim
                    result.scatter_add_(0, buckets, values * signs)
                else:
                    result[offset + start : offset + start + values.numel()] = values
        offset += parameter.numel()
    if not torch.isfinite(norm) or not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite NTK gradients; use float32 and inspect this checkpoint")
    return result.cpu().double().numpy(), norm.item()


def projected_jacobian_rows(output, probes, groups, sketch_dim=8192, sketch_seed=2026):
    """One forward, several shared Rademacher output probes; keep graph only while needed."""
    import torch

    parameters = [p for members in groups.values() for _, p in members]
    rows = {group: [] for group in groups}
    traces = {group: [] for group in groups}
    for probe_index, probe in enumerate(probes):
        gradients = torch.autograd.grad(
            (output * probe).sum(),
            parameters,
            allow_unused=True,
            retain_graph=probe_index < len(probes) - 1,
        )
        offset = 0
        for group, members in groups.items():
            values, trace = compress_gradients(
                members,
                gradients[offset : offset + len(members)],
                0 if group.startswith("prompts/") else sketch_dim,
                sketch_seed,
            )
            rows[group].append(values)
            traces[group].append(trace)
            offset += len(members)
        del gradients
    return rows, traces
