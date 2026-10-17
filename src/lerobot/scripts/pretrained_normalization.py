"""Keep pretrained normalization intact during opt-in continuation experiments."""

from copy import deepcopy


def preserve_pretrained_normalization(processor_kwargs, pretrained_path):
    """Let the processor loader restore saved modes, features, epsilon and tensors.

    Passing even just new stats in an override makes the normalizer ignore its
    saved state. Keep device/tokenizer/rename/relative-action overrides intact.
    The caller must use this with a compatible dataset and pretrained policy.
    """
    if not pretrained_path:
        raise ValueError("preserve_pretrained_normalization requires a pretrained checkpoint")
    kwargs = deepcopy(processor_kwargs)
    kwargs.pop("dataset_stats", None)
    for pipeline, normalizer in (
        ("preprocessor_overrides", "normalizer_processor"),
        ("postprocessor_overrides", "unnormalizer_processor"),
    ):
        if kwargs.get(pipeline):
            kwargs[pipeline].pop(normalizer, None)
    return kwargs
