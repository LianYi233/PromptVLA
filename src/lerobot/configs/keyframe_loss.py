"""Opt-in action-time weighting on the unchanged W0 occurrence-bound recipe."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


@dataclass
class KeyframeLossConfig:
    mode: str = "off"
    # Prepared CPU archive; contains event masks and budget-calibrated schedules, NOT a new sampler.
    archive: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def validate_training(self, cfg: "TrainPipelineConfig") -> None:
        if self.mode not in ("off", "B", "K2", "R2", "Kearly", "Klate"):
            raise ValueError("keyframe_loss.mode must be off/B/K2/R2/Kearly/Klate")
        if not self.enabled:
            if self.archive is not None:
                raise ValueError("keyframe_loss.archive requires an enabled mode")
            return
        policy = cfg.trainable_config
        if (
            self.archive is None
            or not cfg.pdet.enabled
            or cfg.pdet.order_plan is None
            or cfg.pdet.flow_binding != "occurrence"
            or policy.type != "pi05"
            or policy.chunk_size != 50
            or policy.num_vlm_prompt_tokens != 2
            or cfg.seed != 0
            or (cfg.pdet.prompt_seed, cfg.pdet.data_seed) != (1, 0)
            or cfg.pdet.noise_seed != cfg.pdet.time_seed
            or cfg.pdet.noise_seed not in (0, 1)
            or cfg.pdet.audit_steps < cfg.steps
            or cfg.sample_weighting is not None
            or cfg.resume
        ):
            raise ValueError(
                "Keyframe weighting requires native PI05 chunk50 with an imported W0 occurrence plan, "
                "an archive, no sample_weighting and a fresh run (resume is not supported)"
            )
        from lerobot.utils.keyframe_loss import load_archive, validate_order
        from lerobot.utils.pdet_order import load_plan

        arrays, manifest = load_archive(self.archive)
        validate_order(load_plan(cfg.pdet.order_plan), manifest)
        if len(arrays["weight_B"]) != cfg.steps:
            raise ValueError("Keyframe schedule must cover all training updates")
        for actual, expected in (
            (cfg.dataset.root, manifest["score_recipe"]["dataset_root"]),
            (policy.pretrained_path, manifest["score_recipe"]["source"]),
        ):
            if actual is None or Path(actual).resolve() != Path(expected).resolve():
                raise ValueError("Keyframe experiment must use the original scoring dataset/source")
