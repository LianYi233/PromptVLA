"""Explicit native PI05 VLM prompt / full action expert training contracts."""


def is_vlm_adaptation(config):
    return bool(getattr(config, "vlm_prompt_only", False) or getattr(config, "train_action_expert", False))


def validate_vlm_trainables(policy):
    """Reject missing action modules or accidental trainable vision/language weights.

    The causal-LM wrapper's unused vocabulary head is NOT part of the action
    network. Only gemma_expert.model participates in velocity prediction.
    """
    if not is_vlm_adaptation(policy.config):
        raise ValueError("Expected explicit VLM-only or VLM + action expert adaptation")
    model = policy.model
    modules = [model.vlm_prompt_tokens]
    if getattr(policy.config, "train_action_expert", False):
        modules.extend(
            [
                model.paligemma_with_expert.gemma_expert.model,
                model.action_in_proj,
                model.action_out_proj,
                model.time_mlp_in,
                model.time_mlp_out,
            ]
        )
    expected = {id(p) for module in modules for p in module.parameters() if p.numel()}
    actual = {id(p) for p in policy.parameters() if p.requires_grad}
    if expected != actual or not model.vlm_prompt_tokens.weight.requires_grad:
        raise ValueError(
            "Trainable partition mismatch: expected exactly VLM prompt and the configured action network; "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
