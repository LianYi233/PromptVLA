#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import copy
import dataclasses
import logging
import sys
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.cabo import OptimizerStepControl, temporary_optimizer_group_lr_scales
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.training_diagnostics import TrainingDiagnostics, diagnostic_means
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all

_PI05_NEXT_ACTION_PRETRAIN_DIR = "next_action_pretrain"


@torch.no_grad()
def _get_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Return the total L2 norm without modifying any gradients."""
    gradient_norms = []
    parameter_device = None
    for parameter in parameters:
        if parameter_device is None:
            parameter_device = parameter.device
        if parameter.grad is not None:
            gradient_norms.append(
                torch.linalg.vector_norm(parameter.grad.detach(), ord=2, dtype=torch.float32)
            )
    if not gradient_norms:
        return torch.tensor(0.0, device=parameter_device)
    return torch.stack(gradient_norms).norm(2)


def _clip_policy_gradients(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply policy-specific clipping, falling back to global norm clipping.

    A policy-specific ``clip_gradients`` hook owns the complete clipping operation. This lets policies
    such as PI05 clip only their action-side parameters without modifying VLM gradients.
    """
    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if has_method(unwrapped_policy, "clip_gradients"):
        # Custom hooks inspect and modify gradients directly, so gradients must first be unscaled.
        accelerator.unscale_gradients(optimizer)
        grad_norm = _get_gradient_norm(policy.parameters())
        clip_metrics = unwrapped_policy.clip_gradients(accelerator=accelerator)
        return grad_norm, clip_metrics or {}

    if grad_clip_norm > 0:
        return accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm), {}

    # Preserve gradient-norm logging without using clip_grad_norm_(..., inf), which can still write to
    # gradients and therefore is not a true no-op for non-finite values.
    accelerator.unscale_gradients(optimizer)
    return _get_gradient_norm(policy.parameters()), {}


def _prepare_policy_optimizer_step_control(
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    accelerator: "Accelerator",
    grad_norm: torch.Tensor,
) -> OptimizerStepControl:
    """Ask a policy for post-preconditioner optimizer-group scales, if it provides a controller."""
    gradients_finite = torch.isfinite(grad_norm).to(dtype=torch.int64)
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes > 1:
        gradients_finite = accelerator.reduce(gradients_finite, reduction="sum")
    if int(gradients_finite.item()) != num_processes:
        return OptimizerStepControl(
            metrics={
                "optimizer_step/skipped": 1.0,
                "optimizer_step/nonfinite_gradients": 1.0,
            },
            skip_optimizer_step=True,
        )

    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    cabo_active = getattr(getattr(unwrapped_policy, "config", None), "cabo_active", None)
    if cabo_active is False:
        return OptimizerStepControl()
    if not has_method(unwrapped_policy, "compute_optimizer_step_control"):
        return OptimizerStepControl()

    # The main backward pass has already synchronized DDP gradients. The controller only inspects
    # those gradients and reduces aggregate update-norm scalars, so suppress any redundant full
    # gradient synchronization inside the policy hook.
    no_sync = accelerator.no_sync(policy) if has_method(accelerator, "no_sync") else nullcontext()
    with no_sync:
        result = unwrapped_policy.compute_optimizer_step_control(
            batch=batch,
            optimizer=optimizer,
            accelerator=accelerator,
        )
    if isinstance(result, OptimizerStepControl):
        return result

    # Backward compatibility for external policies using the original undocumented two-tuple hook.
    group_scales, metrics = result
    return OptimizerStepControl(group_scales=group_scales or {}, metrics=metrics or {})


def _normalize_policy_loss_for_distributed_training(
    policy: PreTrainedPolicy,
    batch: Any,
    loss: torch.Tensor,
    accelerator: "Accelerator",
) -> torch.Tensor:
    """Reweight a locally normalized loss so DDP optimizes the global valid-element mean.

    DDP averages one gradient per rank. For masked objectives whose number of valid elements differs
    by rank, equally averaging local means is biased. A policy can expose its local denominator via
    ``get_distributed_loss_normalizer``; multiplying by ``world_size * local / global`` makes DDP's
    subsequent gradient average equal the gradient of the global numerator divided by the global
    denominator. The returned value remains graph-connected for ranks whose local denominator is zero.
    """
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        return loss

    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if not has_method(unwrapped_policy, "get_distributed_loss_normalizer"):
        return loss

    local_normalizer = unwrapped_policy.get_distributed_loss_normalizer(batch)
    if not isinstance(local_normalizer, torch.Tensor) or local_normalizer.numel() != 1:
        raise TypeError("get_distributed_loss_normalizer must return a scalar tensor")
    local_normalizer = local_normalizer.detach().to(device=loss.device, dtype=torch.float32).reshape(())
    global_normalizer = accelerator.reduce(local_normalizer, reduction="sum")
    scale = num_processes * local_normalizer / global_normalizer.clamp_min(1)
    return loss * scale.to(dtype=loss.dtype)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
    parameter_monitor=None,
    forward_kwargs=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The fallback maximum norm for policies without a policy-specific clipping hook.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.
        parameter_monitor: Optional sampled gradient/actual-update diagnostics on the main rank.
        forward_kwargs: Optional explicit stochastic flow inputs (PDET experiments only).

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none", **(forward_kwargs or {}))

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch, **(forward_kwargs or {}))

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    if sample_weights is None:
        loss = _normalize_policy_loss_for_distributed_training(
            policy=policy,
            batch=batch,
            loss=loss,
            accelerator=accelerator,
        )

    # Use accelerator's backward method
    accelerator.backward(loss)

    grad_norm, clip_metrics = _clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=grad_clip_norm,
        accelerator=accelerator,
    )
    if clip_metrics:
        if output_dict is None:
            output_dict = {}
        output_dict.update(clip_metrics)

    optimizer_step_control = _prepare_policy_optimizer_step_control(
        policy=policy,
        batch=batch,
        optimizer=optimizer,
        accelerator=accelerator,
        grad_norm=grad_norm,
    )
    if optimizer_step_control.metrics:
        if output_dict is None:
            output_dict = {}
        output_dict.update(optimizer_step_control.metrics)

    if parameter_monitor is not None:
        parameter_monitor.before_step()

    if not optimizer_step_control.skip_optimizer_step:
        # Optimizer step
        with (
            lock if lock is not None else nullcontext(),
            temporary_optimizer_group_lr_scales(optimizer, optimizer_step_control.group_scales),
        ):
            optimizer.step()
    else:
        # FP16 unscaling creates per-optimizer GradScaler state that is normally finalized by
        # AcceleratedOptimizer.step(). The controller deliberately skips that call, so finalize the
        # scaler explicitly; otherwise the next iteration's unscale_ reports that it was already called.
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            scaler.update()

    if parameter_monitor is not None:
        if output_dict is None:
            output_dict = {}
        output_dict.update(parameter_monitor.after_step())
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None and not optimizer_step_control.skip_optimizer_step:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if not optimizer_step_control.skip_optimizer_step and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


def _pi05_next_action_pretraining_active(cfg: TrainPipelineConfig) -> bool:
    """Return whether this run needs integrated PI0.5 frozen-VLM Stage 1."""
    policy_cfg = cfg.policy
    return bool(
        not cfg.resume
        and not cfg.is_reward_model_training
        and policy_cfg is not None
        and policy_cfg.type == "pi05"
        and getattr(policy_cfg, "next_action_pretraining_active", False)
    )


def _make_pi05_next_action_pretraining_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    """Build an isolated frozen-VLM Stage 1 without mutating the formal flow config."""
    from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05TrainingStage

    if not _pi05_next_action_pretraining_active(cfg) or not isinstance(cfg.policy, PI05Config):
        raise ValueError("Integrated PI0.5 Stage 1 requires a non-resumed flow run")
    if cfg.output_dir is None:
        raise ValueError("output_dir must be resolved before building the PI0.5 pretraining stage")
    if cfg.policy.use_peft or cfg.peft is not None:
        raise ValueError(
            "Integrated PI0.5 Stage 1 does not support PEFT. Use a standard PI0.5 "
            "base/flow checkpoint as --policy.path and run prompt training without a PEFT wrapper."
        )

    pretrain_cfg = copy.deepcopy(cfg)
    pretrain_steps = cfg.policy.next_action_pretrain_steps
    bridge_steps = cfg.policy.next_action_bridge_steps
    if bridge_steps > pretrain_steps:
        raise ValueError(
            "next_action_bridge_steps cannot exceed next_action_pretrain_steps for integrated "
            f"Stage 1, got {bridge_steps} > {pretrain_steps}"
        )
    pretrain_cfg.policy = dataclasses.replace(
        pretrain_cfg.policy,
        training_stage=PI05TrainingStage.NEXT_ACTION,
        next_action_pretrain_steps=0,
        # Both Stage-1 objectives use this fixed recipe, independently of any formal-flow optimizer
        # overrides in the user's one-command invocation.
        optimizer_lr=2.5e-5,
        optimizer_betas=(0.9, 0.95),
        optimizer_eps=1e-8,
        optimizer_weight_decay=0.01,
        optimizer_grad_clip_norm=0.0,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
        scheduler_decay_lr=1e-5,
        push_to_hub=False,
    )
    pretrain_cfg.steps = pretrain_steps
    pretrain_cfg.output_dir = (
        cfg.output_dir.parent / f"{cfg.output_dir.name}_{_PI05_NEXT_ACTION_PRETRAIN_DIR}"
    )
    pretrain_cfg.job_name = f"{cfg.job_name}_next_action_pretrain"
    pretrain_cfg.resume = False
    pretrain_cfg.checkpoint_path = None
    pretrain_cfg.env = None
    pretrain_cfg.env_eval_freq = 0
    pretrain_cfg.peft = None
    pretrain_cfg.sample_weighting = None
    pretrain_cfg.save_checkpoint = True
    pretrain_cfg.save_freq = pretrain_steps
    # Formal-flow milestones must not suppress the transfer checkpoint or leak into Stage 1.
    pretrain_cfg.save_steps = [pretrain_steps] if cfg.save_steps is not None else None
    pretrain_cfg.save_checkpoint_to_hub = False
    pretrain_cfg.wandb = dataclasses.replace(pretrain_cfg.wandb, run_id=None)
    pretrain_cfg.job = dataclasses.replace(pretrain_cfg.job, target="local")
    pretrain_cfg.use_policy_training_preset = True
    pretrain_cfg.optimizer = pretrain_cfg.policy.get_optimizer_preset()
    pretrain_cfg.scheduler = pretrain_cfg.policy.get_scheduler_preset()
    return pretrain_cfg


def _pi05_next_action_pretrained_model_dir(pretrain_cfg: TrainPipelineConfig) -> Path:
    """Return the deterministic final model directory produced by the first stage."""
    if pretrain_cfg.output_dir is None:
        raise ValueError("The PI0.5 pretraining output_dir is not configured")
    checkpoint_dir = get_step_checkpoint_dir(
        pretrain_cfg.output_dir,
        pretrain_cfg.steps,
        pretrain_cfg.steps,
    )
    return checkpoint_dir / PRETRAINED_MODEL_DIR


def _save_pi05_ntk_initial_snapshot(
    cfg, policy, optimizer, scheduler, preprocessor, postprocessor, accelerator, step
) -> None:
    """Save actual initialized prompts before the first update, collectively under FSDP."""
    from accelerate import DistributedType

    if not (
        getattr(cfg.policy, "ntk_save_stage_snapshots", False)
        and cfg.save_checkpoint
        and not cfg.resume
        and step == 0
    ):
        return
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict, optim_state_dict = (
        gather_fsdp_state_dicts(policy, optimizer) if is_fsdp else (None, None)
    )
    if accelerator.is_main_process:
        logging.info("Saving exact step-0 model for longitudinal NTK analysis")
        save_checkpoint(
            checkpoint_dir=get_step_checkpoint_dir(cfg.output_dir, cfg.steps, 0),
            step=0,
            cfg=cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            num_processes=accelerator.num_processes,
            batch_size=cfg.batch_size,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
        )
    accelerator.wait_for_everyone()


def _pi05_stage1_bridge_start_step(cfg: TrainPipelineConfig) -> int | None:
    """Return the zero-based Stage-1 update where observation-conditioned flow begins."""
    policy_cfg = cfg.policy
    if (
        policy_cfg is None
        or policy_cfg.type != "pi05"
        or getattr(policy_cfg, "training_stage", None) != "next_action"
    ):
        return None
    bridge_steps = int(getattr(policy_cfg, "next_action_bridge_steps", 0))
    if bridge_steps > cfg.steps:
        raise ValueError(
            "next_action_bridge_steps cannot exceed the Stage-1 training steps, "
            f"got {bridge_steps} > {cfg.steps}"
        )
    return cfg.steps - bridge_steps if bridge_steps > 0 else None


def _set_policy_training_progress(policy, accelerator: "Accelerator", *, step: int, total_steps: int) -> None:
    """Pass the checkpoint-aware global update index to policies with runtime curricula."""
    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if has_method(unwrapped_policy, "set_training_progress"):
        unwrapped_policy.set_training_progress(step=step, total_steps=total_steps)


def _make_training_accelerator(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator | None" = None,
) -> "Accelerator":
    """Create the one Accelerator shared by both PI0.5 objectives in an integrated run."""
    if accelerator is not None:
        return accelerator

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Accelerate auto-detects the device based on the available hardware and ignores policy.device.
    force_cpu = cfg.trainable_config.device == "cpu"
    policy_dtype = getattr(cfg.trainable_config, "dtype", None)
    mixed_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(policy_dtype)
    return Accelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )


def _run_pi05_next_action_pretraining(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator",
) -> None:
    """Run Stage 1, then point the untouched formal config at its transferable weights."""
    pretrain_cfg = _make_pi05_next_action_pretraining_config(cfg)
    bridge_steps = pretrain_cfg.policy.next_action_bridge_steps
    logging.info(
        "Starting integrated PI0.5 Stage 1 with %d action-only flow-inpainting updates followed by %d "
        "observation-conditioned full-flow bridge updates with the VLM frozen; formal flow training will "
        "start automatically afterwards.",
        pretrain_cfg.steps - bridge_steps,
        bridge_steps,
    )
    _train_single_stage(pretrain_cfg, accelerator, enable_pi05_stage1_bridge=True)
    accelerator.wait_for_everyone()
    pretrained_model_dir = _pi05_next_action_pretrained_model_dir(pretrain_cfg)
    checkpoint_visible = pretrained_model_dir.is_dir()
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes > 1:
        visible_count = accelerator.reduce(
            torch.tensor(int(checkpoint_visible), device=accelerator.device),
            reduction="sum",
        )
        checkpoint_visible = int(visible_count.item()) == num_processes
    if not checkpoint_visible:
        raise RuntimeError(
            "Integrated PI0.5 Stage 1 finished without producing its final "
            f"checkpoint at {pretrained_model_dir} on every rank. Distributed training requires "
            "output_dir and its sibling pretraining directory to be on a filesystem shared by all ranks."
        )
    accelerator.free_memory()
    cfg.policy.pretrained_path = pretrained_model_dir
    cfg.policy.pretrained_revision = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(
        "PI0.5 Stage 1 complete; rebuilding the model, optimizer, and scheduler for "
        "formal flow training with save_steps=%s, save_freq=%d (final step always saved).",
        cfg.save_steps,
        cfg.save_freq,
    )


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate.utils import DistributedType

    cfg.validate()
    accelerator = _make_training_accelerator(cfg, accelerator)

    try:
        # Process-group construction is not a barrier. In particular, rank 0 must
        # not let WandBLogger create output_dir while a slower rank is still in
        # cfg.validate() checking that a fresh run's directory does not exist.
        accelerator.wait_for_everyone()

        if (
            cfg.prompt_rank.freq > 0
            or cfg.pdet.enabled
            or cfg.vlm_prompt_init_path is not None
            or cfg.vlm_prompt_export_path is not None
        ) and accelerator.distributed_type not in (
            DistributedType.NO,
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_CPU,
        ):
            raise NotImplementedError(
                "prompt diagnostics/initialization support unsharded single-device/DDP only"
            )
        if cfg.piper_eval.enabled and accelerator.distributed_type not in (
            DistributedType.NO,
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_CPU,
        ):
            raise NotImplementedError("piper_eval supports unsharded single-device/DDP training only")
        if cfg.cabo_active and accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError(
                "PI0.5 CABO currently supports single-device and DDP training only; FSDP parameter "
                "sharding is not yet supported."
            )

        if _pi05_next_action_pretraining_active(cfg):
            _run_pi05_next_action_pretraining(cfg, accelerator)

        return _train_single_stage(cfg, accelerator)
    finally:
        accelerator.end_training()


def _train_single_stage(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator",
    *,
    enable_pi05_stage1_bridge: bool = False,
):
    """Train exactly one already-validated objective with freshly created training state."""
    from accelerate.utils import DistributedType

    active_cfg = cfg.trainable_config
    pi05_stage1_bridge_start_step = _pi05_stage1_bridge_start_step(cfg) if enable_pi05_stage1_bridge else None
    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process and cfg.vlm_prompt_export_path is None:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.vlm_prompt_export_path is not None:
        from lerobot.utils.prompt_initialization import export_constructed_prompt

        # Use exactly the normal seeded dataset/model construction above. No optimizer,
        # DDP wrapping, rank diagnostics, training updates or full checkpoint writes follow.
        return export_constructed_prompt(policy, cfg, accelerator)

    if cfg.vlm_prompt_init_path is not None and not cfg.resume:
        from lerobot.utils.prompt_initialization import apply_initial_prompt, save_import_record
        from lerobot.utils.prompt_rank import collective_call

        # Keep seeded construction above intact, including its random draws. Only copy prompt values.
        record = collective_call(accelerator, lambda: apply_initial_prompt(policy, cfg.vlm_prompt_init_path))
        digest = torch.tensor(
            list(bytes.fromhex(record["loaded_prompt_sha256"])), device=device, dtype=torch.uint8
        )
        digests = accelerator.gather(digest).reshape(accelerator.num_processes, -1)
        if not (digests == digests[0]).all().item():
            raise ValueError("Imported prompt initialization differs across DDP ranks")
        collective_call(
            accelerator,
            lambda: save_import_record(policy, record, cfg.output_dir, cfg.seed) if is_main_process else None,
        )
        if is_main_process:
            logging.info(
                "PROMPT_INIT source_training_seed=%s training_seed=%s sha256=%s (training RNG unchanged)",
                record["source_training_seed"],
                cfg.seed,
                record["loaded_prompt_sha256"],
            )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        tokenizer_name = getattr(active_cfg, "tokenizer_name", None)
        if tokenizer_name:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_name}
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.preserve_pretrained_normalization:
        from lerobot.scripts.pretrained_normalization import preserve_pretrained_normalization

        processor_kwargs = preserve_pretrained_normalization(processor_kwargs, processor_pretrained_path)

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        try:
            step, optimizer, lr_scheduler = load_training_state(
                cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
            )
        except ValueError as exc:
            incompatible_groups = "parameter group" in str(exc).lower() or "param_groups" in str(exc).lower()
            if cfg.cabo_active and incompatible_groups:
                raise ValueError(
                    "An incompatible optimizer checkpoint cannot be resumed with CABO enabled because "
                    "CABO uses named VLM prompt and action prompt parameter groups. Start a "
                    "new run from the checkpoint's model weights, or resume a checkpoint that was "
                    "already trained with this CABO parameter-group layout."
                ) from exc
            raise

    if cfg.cabo_active and has_method(policy, "validate_optimizer_step_control"):
        policy.validate_optimizer_step_control(optimizer)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # TRAINABLE_PARAMETER_REPORT_V1
    if is_main_process:
        _parameter_counts = {"vlm_prompt": 0, "action_prompt": 0, "action_projection": 0, "other": 0}
        logging.info("========== Trainable parameter report ==========")
        logging.info("Training stage: %s", getattr(cfg.policy, "training_stage", "unknown"))
        for _name, _parameter in policy.named_parameters():
            if not _parameter.requires_grad:
                continue
            _count = _parameter.numel()
            if "vlm_prompt_tokens" in _name.split("."):
                _category = "vlm_prompt"
            elif "prompt_tokens" in _name.split("."):
                _category = "action_prompt"
            elif any(part in _name.split(".") for part in ("action_in_proj", "action_out_proj")):
                _category = "action_projection"
            else:
                _category = "other"
            _parameter_counts[_category] += _count
            logging.info(
                "TRAINABLE %s | shape=%s | params=%s",
                _name,
                tuple(_parameter.shape),
                f"{_count:,}",
            )
        for _category, _count in _parameter_counts.items():
            logging.info("%s: %s (%.6fM)", _category, f"{_count:,}", _count / 1e6)
        logging.info(
            "Trainable total: %s (%.6fM) | All parameters: %s | Trainable ratio: %.6f%%",
            f"{num_learnable_params:,}",
            num_learnable_params / 1e6,
            f"{num_total_params:,}",
            100 * num_learnable_params / max(1, num_total_params),
        )
        logging.info("================================================")

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    training_dataset = dataset
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.pdet.data_seed if cfg.pdet.enabled else (cfg.seed if cfg.seed is not None else 0),
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0 and cfg.pdet.order_plan is None:
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written with "
                    f"batch_size={saved_batch_size}. The data order resumes at the right epoch/offset, "
                    "but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_num_processes)
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
        if cfg.pdet.order_plan is not None:
            from lerobot.utils.pdet_order import PlannedDataset, PlannedSampler, load_plan

            sampler = PlannedSampler(load_plan(cfg.pdet.order_plan), start_step=step)
            training_dataset = PlannedDataset(dataset)
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    # Worker startup must not consume the global RNG or follow D (D is sample order only).
    worker_generator = torch.Generator().manual_seed(cfg.seed) if cfg.pdet.enabled else None
    dataloader = torch.utils.data.DataLoader(
        training_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        generator=worker_generator,
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    dl_iter = cycle(dataloader)

    pdet_streams = None
    if cfg.pdet.enabled:
        from lerobot.utils.pdet import PDETStreams
        from lerobot.utils.prompt_rank import collective_call

        pdet_streams = collective_call(
            accelerator, lambda: PDETStreams(cfg, accelerator.unwrap_model(policy), accelerator)
        )

    policy.train()

    parameter_monitor = None
    action_evaluator = None
    if cfg.piper_eval.enabled:
        from lerobot.utils.piper_training_eval import (
            ActionParameterMonitor,
            PiperActionEvaluator,
            on_main_process,
        )

        if cfg.piper_eval.update_freq > 0 and is_main_process:
            parameter_monitor = ActionParameterMonitor(accelerator.unwrap_model(policy))
        if cfg.piper_eval.freq > 0:
            splits = {"train": dataset}
            if eval_dataset is not None:
                splits["val"] = eval_dataset
            action_evaluator = on_main_process(
                accelerator,
                lambda: PiperActionEvaluator(
                    splits, active_cfg, preprocessor, postprocessor, cfg.piper_eval, cfg.output_dir, device
                ),
            )

    if cfg.pdet.enabled and cfg.pdet.diagnostics and is_main_process:
        from lerobot.utils.pdet import PromptUpdateMonitor

        parameter_monitor = PromptUpdateMonitor(accelerator.unwrap_model(policy))

    def log_action_evaluation(completed_step):
        # All ranks participate in error propagation/synchronization; only rank 0 predicts.
        if is_main_process:
            logging.info(
                "PIPER_ACTION_EVAL starting step=%d (full denoising on fixed observations)", completed_step
            )
        scores = on_main_process(
            accelerator,
            lambda: action_evaluator.evaluate(
                accelerator.unwrap_model(policy), completed_step, accelerator.autocast
            ),
        )
        if is_main_process:
            key = f"action_train/h{cfg.piper_eval.execution_steps}/joint_mae_deg"
            logging.info("PIPER_ACTION_EVAL step=%d %s=%.4f", completed_step, key, scores[key])
            if wandb_logger:
                wandb_logger.log_dict(scores, step=completed_step, mode="eval")

    if cfg.piper_eval.freq > 0:
        log_action_evaluation(step)

    prompt_rank_monitor = None
    if cfg.prompt_rank.freq > 0:
        from lerobot.utils.prompt_rank import PromptRankMonitor, collective_call

        prompt_rank_monitor = collective_call(
            accelerator,
            lambda: PromptRankMonitor(
                dataset, preprocessor, active_cfg, cfg.prompt_rank, cfg.output_dir, device
            ),
        )
        prompt_rank_monitor.verify_across_ranks(accelerator)
        collective_call(
            accelerator, lambda: prompt_rank_monitor.save_protocol(step) if is_main_process else None
        )

    def log_prompt_rank(completed_step):
        if is_main_process:
            logging.info("PROMPT_RANK starting step=%d (fixed velocity Jacobian probes)", completed_step)
        scores = prompt_rank_monitor.measure(accelerator.unwrap_model(policy), completed_step, accelerator)
        if is_main_process:
            logging.info(
                "PROMPT_RANK step=%d J-erank=%.3f NTK-erank=%.3f rank95=%d ceiling=%d time=%.1fs",
                completed_step,
                scores["jacobian_erank"],
                scores["ntk_erank"],
                scores["rank95_energy"],
                scores["rank_ceiling"],
                scores["probe_seconds"],
            )
            if wandb_logger:
                wandb_logger.log_dict(
                    {f"prompt_rank/{key}": value for key, value in scores.items()},
                    step=completed_step,
                    mode="eval",
                )

    if prompt_rank_monitor is not None:
        log_prompt_rank(step)

    pi05_diagnostics_enabled = not cfg.is_reward_model_training and active_cfg.type == "pi05"
    diagnostics = None
    if is_main_process and pi05_diagnostics_enabled:
        diagnostics = TrainingDiagnostics(
            Path(cfg.output_dir) / "training_diagnostics.jsonl",
            {
                "training_stage": str(active_cfg.training_stage),
                "train_action_projections": getattr(active_cfg, "train_action_projections", False),
                "trainable_parameters": num_learnable_params,
                "action_names": dataset.meta.features["action"].get("names"),
                "num_processes": accelerator.num_processes,
                "policy_metrics_scope": "rank0 per-update values; train.loss uses existing DDP reductions",
            },
        )
        logging.info("Local PI05 diagnostics: %s", diagnostics.path)

    ntk_snapshots = getattr(cfg.policy, "ntk_save_stage_snapshots", False)
    _save_pi05_ntk_initial_snapshot(
        cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor, accelerator, step
    )

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        sample_indices = batch.get("index") if pdet_streams is not None else None
        # Capture identities before processor packing; never expose experiment keys to the model.
        occurrence_ids = batch.pop("_pdet_occurrence_id", None)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        forward_kwargs = None
        if pdet_streams is not None:

            def draw_flow(
                batch=batch, step=step, sample_indices=sample_indices, occurrence_ids=occurrence_ids
            ):
                return pdet_streams.draw(
                    accelerator.unwrap_model(policy), batch, step, sample_indices, occurrence_ids
                )

            # Propagate audit I/O failures before any rank enters backward collectives.
            forward_kwargs = (
                collective_call(accelerator, draw_flow) if step < cfg.pdet.audit_steps else draw_flow()
            )

        if enable_pi05_stage1_bridge:
            _set_policy_training_progress(policy, accelerator, step=step, total_steps=cfg.steps)
        if is_main_process and step == pi05_stage1_bridge_start_step:
            logging.info(
                "Switching PI0.5 Stage 1 to observation-conditioned full-flow for the final %d updates; "
                "the VLM remains frozen and the existing optimizer and scheduler continue unchanged.",
                cfg.steps - step,
            )

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
            forward_kwargs=forward_kwargs,
            parameter_monitor=(
                parameter_monitor
                if (cfg.pdet.enabled and cfg.pdet.diagnostics)
                or (
                    cfg.piper_eval.update_freq > 0
                    and ((step + 1) % cfg.piper_eval.update_freq == 0 or step + 1 == cfg.steps)
                )
                else None
            ),
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        if diagnostics is not None:
            diagnostics.update(output_dict)
        is_log_step = (cfg.log_freq > 0 and step % cfg.log_freq == 0) or (
            pi05_diagnostics_enabled and step == cfg.steps
        )
        is_saving_step = cfg.should_save_checkpoint(step)
        if ntk_snapshots and step == pi05_stage1_bridge_start_step:
            # `step` counts completed updates: 750 here is after priming, before bridge update 1.
            is_saving_step = True
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                diagnostic_record = None
                if diagnostics is not None:
                    diagnostic_record = diagnostics.write(step, train_tracker.to_dict())
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if diagnostic_record:
                        wandb_log_dict.update(diagnostic_means(diagnostic_record))
                    elif output_dict:
                        wandb_log_dict.update(output_dict)
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.piper_eval.freq > 0 and (
            step % cfg.piper_eval.freq == 0 or step == cfg.steps or step == pi05_stage1_bridge_start_step
        ):
            log_action_evaluation(step)

        if prompt_rank_monitor is not None and (
            step % cfg.prompt_rank.freq == 0 or step == cfg.steps or step in cfg.prompt_rank.extra_steps
        ):
            log_prompt_rank(step)

        if is_eval_step:
            policy.eval()
            # Evaluate the objective used by the next update at an exact curriculum boundary. At
            # the final update, retain the final objective instead of passing an out-of-range step.
            if enable_pi05_stage1_bridge:
                _set_policy_training_progress(
                    policy,
                    accelerator,
                    step=min(step, cfg.steps - 1),
                    total_steps=cfg.steps,
                )
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    loss = _normalize_policy_loss_for_distributed_training(
                        policy=policy,
                        batch=eval_batch,
                        loss=loss,
                        accelerator=accelerator,
                    )
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model, dataset_meta=dataset.meta)
            else:
                unwrapped_model.push_model_to_hub(cfg, state_dict=model_state_dict, dataset_meta=dataset.meta)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    if wandb_logger:
        wandb_logger.finish()
    # Accelerator lifecycle is owned by train(): Stage 1 must leave it alive for the fresh flow model.
    accelerator.wait_for_everyone()


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
