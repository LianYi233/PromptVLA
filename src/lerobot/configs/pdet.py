"""Independent random factors for the native VLM-only replication experiment."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


@dataclass
class PDETConfig:
    enabled: bool = False
    # P labels an imported, verified step-zero tensor; it does not reseed model construction.
    prompt_seed: int = 0
    data_seed: int = 0
    noise_seed: int = 0
    time_seed: int = 0
    audit_steps: int = 8
    # Optional finite experiment plan. This selects a NEW occurrence-keyed flow protocol.
    order_plan: str | None = None
    diagnostics: bool = False

    def validate(self) -> None:
        if not self.enabled and (self.order_plan is not None or self.diagnostics):
            raise ValueError("PDET order plans/diagnostics require pdet.enabled=true")
        for name in ("prompt_seed", "data_seed", "noise_seed", "time_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"pdet.{name} must be an integer in [0, 2**32)")
        if type(self.audit_steps) is not int or self.audit_steps < 1:
            raise ValueError("pdet.audit_steps must be positive")

    @property
    def label(self) -> str:
        return f"P{self.prompt_seed}-D{self.data_seed}-E{self.noise_seed}-T{self.time_seed}"

    def validate_training(self, cfg: "TrainPipelineConfig") -> None:
        self.validate()
        if not self.enabled:
            return
        policy = cfg.trainable_config
        if (
            cfg.is_reward_model_training
            or policy.type != "pi05"
            or not getattr(policy, "vlm_prompt_only", False)
            or policy.num_prompt_tokens != 0
            or policy.compile_model
            or policy.use_peft
            or cfg.peft is not None
            or policy.training_stage != "flow"
            or policy.next_action_pretrain_steps != 0
            or policy.next_action_bridge_steps != 0
            or cfg.dataset.streaming
            or cfg.dataset.image_transforms.enable
            or cfg.dataset.eval_split != 0
            or cfg.env_eval_freq != 0
            or cfg.eval_steps != 0
            or cfg.vlm_prompt_export_path is not None
            or cfg.vlm_prompt_init_path is None
            or (self.order_plan is not None and cfg.dataset.episodes is not None)
            or type(cfg.seed) is not int
            or not 0 <= cfg.seed < 2**32
        ):
            raise ValueError(
                "PDET requires seeded native VLM-only flow with imported P, no compile/PEFT/"
                "priming/bridge, and a non-streaming dataset without augmentation or online evaluation"
            )
        if self.order_plan is not None:
            from lerobot.utils.pdet_order import load_plan

            plan = load_plan(self.order_plan)
            if plan["manifest"]["data_seed"] != self.data_seed:
                raise ValueError("PDET D must match the fixed task/pool plan seed")
            if plan["manifest"]["steps"] != cfg.steps:
                raise ValueError("Order plan must cover exactly the configured training steps")
        if not cfg.resume:
            from lerobot.utils.prompt_initialization import read_initial_prompt

            _, metadata = read_initial_prompt(cfg.vlm_prompt_init_path, policy.num_vlm_prompt_tokens)
            if metadata["source_training_seed"] != self.prompt_seed or metadata.get(
                "source_prompt_init_path"
            ):
                raise ValueError(
                    "P label must match the original initialization seed, not a swapped run's seed"
                )
