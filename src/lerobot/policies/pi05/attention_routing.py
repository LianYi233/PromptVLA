"""Partition PI0.5 keys exhaustively; never normalize attention within a modality."""

import numpy as np


def build_key_groups(image_lengths, language_mask, vlm_prompts, action_prompts, chunk_size, executed):
    """Return one group ID per key in embed_prefix + embed_suffix order.

    Keep empty groups (e.g. action_prompt for vlm_only) so schemas remain comparable.
    Camera groups include padded cameras; their actual visibility is measured from
    the attention mask. Language padding gets an explicit zero-mass control group.
    """
    names, groups = [], []

    def add(name, count):
        groups.extend([len(names)] * count)
        names.append(name)

    for index, length in enumerate(image_lengths):
        add(f"camera_{index}", length)
    language_mask = np.asarray(language_mask, dtype=bool)
    language_id = len(names)
    names.extend(["language_state", "language_padding"])
    groups.extend(np.where(language_mask, language_id, language_id + 1).tolist())
    add("vlm_prompt", vlm_prompts)
    add("action_prompt", action_prompts)
    executed = min(executed, chunk_size)
    add("action_executed", executed)
    add("action_future", chunk_size - executed)
    return names, np.asarray(groups, dtype=np.int64)


def summarize_routing(head_probabilities, mask, start, stop, group_ids, group_names):
    """Head probabilities are [1,H,K], already averaged over selected queries."""
    import torch

    if head_probabilities.shape[0] != 1 or head_probabilities.shape[-1] != len(group_ids):
        raise ValueError("Routing key layout does not match captured attention")
    ids = torch.as_tensor(group_ids, device=head_probabilities.device)
    selected_mask = mask[..., start:stop, : len(group_ids)]
    allowed = selected_mask if selected_mask.dtype == torch.bool else selected_mask > -1e4
    # Mean visible key counts per selected query (and mask head, normally singleton).
    visibility = allowed.float().mean(dim=(0, 1, 2))
    masses = torch.stack(
        [head_probabilities[0, :, ids == i].sum(dim=-1) for i in range(len(group_names))], dim=-1
    )
    counts = torch.stack([visibility[ids == i].sum() for i in range(len(group_names))])
    if not torch.allclose(masses.sum(dim=-1), torch.ones_like(masses[:, 0]), atol=2e-6, rtol=0):
        raise ValueError("Routing groups do not sum to one for every head")
    padding = group_names.index("language_padding")
    if torch.any(masses[:, padding] != 0):
        raise ValueError("Masked language padding received nonzero attention")
    return (
        masses.cpu().numpy(),
        counts.cpu().numpy(),
        head_probabilities[0].mean(dim=0).cpu().numpy(),
    )
