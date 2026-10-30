"""Opt-in instrumentation of the existing trainer; no legacy source files edited.

Only dataset/processor return values and completed update boundaries are observed.
The original update_policy owns every backward, optimizer and scheduler operation.
Hooks exist only inside this process and are restored even on failure.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from examples.analysis.curriculum_study.core import (  # noqa: E402
    SCHEMA, digest, json_hash, load_plan, lr_value, read_json, write_json,
)


def collective(accelerator, callback):
    """Propagate local Python failures before entering the following collective."""
    import torch
    value, error = None, None
    try:
        value = callback()
    except Exception as exc:
        error = exc
        logging.exception("Curriculum study failed on rank %s", accelerator.process_index)
    failures = accelerator.reduce(torch.tensor(int(error is not None), device=accelerator.device),
                                  reduction="sum")
    if failures.item():
        raise RuntimeError(f"Curriculum study failed: {error or 'see peer-rank traceback'}") from error
    return value


@contextlib.contextmanager
def preserve_rng(device):
    import torch
    py, np_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


@contextlib.contextmanager
def inference_flags(policy):
    """No calls to PI05.train()/eval(), which also enforce parameter partitions."""
    flags = [(module, module.training) for module in policy.modules()]
    try:
        for module, _ in flags:
            module.training = False
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def tensor_fingerprint(samples, noise):
    import torch
    h = hashlib.sha256()
    for sample in [*samples, {"noise": noise}]:
        for key, value in sorted(sample.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Probe must contain tensors only: {key}")
            x = value.detach().cpu().contiguous()
            h.update(str((key, str(x.dtype), tuple(x.shape))).encode())
            h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class FixedProbe:
    """Fixed preprocessed observations + fixed noise/time, with valid-target masks."""
    def __init__(self, dataset, preprocessor, policy, protocol, output, device, *, collate_fn=None):
        import torch
        if collate_fn is None:
            from lerobot.utils.collate import lerobot_collate_fn
            collate_fn = lerobot_collate_fn
        self.protocol = protocol
        self.device = torch.device(device)
        self.directory = Path(output) / "curriculum_probe"
        self.samples = []
        transforms = dataset.image_transforms
        with preserve_rng(device):
            processor = copy.deepcopy(preprocessor)
            dataset.clear_image_transforms()
            try:
                for identity in protocol["samples"]:
                    item = dataset[identity["dataset_index"]]
                    for key in ("index", "task_index", "episode_index", "frame_index"):
                        if key not in item or int(item[key]) != identity[key]:
                            raise ValueError(f"Probe dataset identity mismatch: {identity} / {key}")
                    batch = collate_fn([item])
                    for camera in dataset.meta.camera_keys:
                        if batch[camera].dtype == torch.uint8:
                            batch[camera] = batch[camera].float() / 255
                    batch = processor(batch)
                    keep = set(policy.config.image_features) | {
                        "action", "action_is_pad", "observation.language.tokens",
                        "observation.language.attention_mask"}
                    self.samples.append({k: v.detach().cpu().clone() for k, v in batch.items() if k in keep})
            finally:
                dataset.set_image_transforms(transforms)
        self.horizon = policy.config.chunk_size
        self.action_dim = policy.config.output_features["action"].shape[0]
        self.max_action_dim = policy.config.max_action_dim
        generator = torch.Generator().manual_seed(protocol["seed"])
        self.noise = torch.randn(len(self.samples), protocol["noise_repeats"], 1,
                                 self.horizon, self.max_action_dim, generator=generator)
        self.processed_id = tensor_fingerprint(self.samples, self.noise)
        self.manifest = {"protocol": SCHEMA, "probe": protocol,
                         "processed_inputs_id": self.processed_id,
                         "dtype": policy.config.dtype, "action_dim": self.action_dim,
                         "max_action_dim": self.max_action_dim, "horizon": self.horizon,
                         "scope": protocol["scope"]}

    def save(self, accelerator):
        import torch
        value = torch.tensor(list(bytes.fromhex(self.processed_id)), dtype=torch.uint8,
                             device=self.device)
        all_ids = accelerator.gather(value).reshape(accelerator.num_processes, -1)
        if not torch.equal(all_ids, all_ids[:1].expand_as(all_ids)):
            raise ValueError("Processed probe inputs/noise differ across ranks")
        def persist():
            if not accelerator.is_main_process:
                return
            self.directory.mkdir(parents=True, exist_ok=False)
            write_json(self.directory / "manifest.json", self.manifest, exclusive=True)
            torch.save({"samples": self.samples, "noise": self.noise}, self.directory / "fixed_probe.pt")
        collective(accelerator, persist)

    def measure(self, policy, step, accelerator):
        import torch
        times = self.protocol["timesteps"]
        repeats = self.protocol["noise_repeats"]
        def local_measure():
            totals = torch.zeros((len(self.samples), len(times), repeats, 2),
                                 dtype=torch.float64, device=self.device)
            with preserve_rng(self.device), inference_flags(policy), torch.no_grad():
                for i in range(accelerator.process_index, len(self.samples), accelerator.num_processes):
                    batch = {k: v.to(self.device) for k, v in self.samples[i].items()}
                    actions = policy.prepare_action(batch)
                    valid = ~batch.get("action_is_pad", torch.zeros((1, self.horizon),
                                        dtype=torch.bool, device=self.device)).bool()
                    count = int(valid.sum().item()) * self.action_dim
                    if not count:
                        raise ValueError(f"Probe {i} has no valid action target")
                    for ti, t in enumerate(times):
                        for r in range(repeats):
                            noise = self.noise[i, r].to(self.device)
                            with accelerator.autocast():
                                images, masks = policy._preprocess_images(batch)
                                velocity = policy.model.predict_velocity(
                                    images, masks, batch["observation.language.tokens"],
                                    batch["observation.language.attention_mask"],
                                    t * noise + (1 - t) * actions,
                                    torch.tensor([t], device=self.device, dtype=torch.float32))
                            squared = (velocity[..., :self.action_dim].float() -
                                       (noise - actions)[..., :self.action_dim].float()).square()
                            error = squared[valid].double().sum()
                            if not torch.isfinite(error):
                                raise FloatingPointError(f"Nonfinite probe loss at sample {i}")
                            totals[i, ti, r, 0] = error
                            totals[i, ti, r, 1] = count
            return totals
        local = collective(accelerator, local_measure)
        total = accelerator.reduce(local, reduction="sum").cpu().numpy()
        def persist():
            if not accelerator.is_main_process:
                return None
            rows = []
            for i, identity in enumerate(self.protocol["samples"]):
                for ti, t in enumerate(times):
                    for r in range(repeats):
                        rows.append({**identity, "time": t, "noise_repeat": r,
                                     "squared_error_sum": float(total[i, ti, r, 0]),
                                     "valid_coordinates": int(total[i, ti, r, 1])})
            record = {"step": int(step), "probe_id": self.protocol["probe_id"],
                      "processed_inputs_id": self.processed_id, "samples": rows}
            write_json(self.directory / f"step_{step:06d}.json", record, exclusive=True)
            from examples.analysis.curriculum_study.analysis import aggregate
            summary = aggregate(rows)
            logging.info("CURRICULUM_PROBE step=%d %s", step, summary)
            return summary
        return collective(accelerator, persist)


def check_spec(path):
    spec = read_json(path)
    if spec.get("protocol") != SCHEMA or spec.get("spec_id") != json_hash(
            {k: v for k, v in spec.items() if k != "spec_id"}):
        raise ValueError("Invalid study spec/checksum")
    for filename, expected in spec["input_sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"Study input changed: {filename}")
    for filename, expected in spec["code_sha256"].items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f"Study instrumentation changed: {filename}; prepare a new study")
    weights = spec["weight_inventory"]
    source = Path(weights["path"])
    if (str(source.resolve()), source.stat().st_size, source.stat().st_mtime_ns) != (
            weights["resolved"], weights["size"], weights["mtime_ns"]):
        raise ValueError("Source model file inventory changed; prepare a new study")
    plan = load_plan(spec["plan"])
    probe = read_json(spec["probe"])
    if plan["manifest"]["instance_pool_sha256"] != probe["pool_sha256"] or np.intersect1d(
            plan["episode_index"], probe["heldout_episodes"]).size:
        raise ValueError("Probe overlap or different occurrence pool")
    if probe["probe_id"] != json_hash({k: v for k, v in probe.items() if k != "probe_id"}):
        raise ValueError("Invalid probe checksum")
    return spec, probe


@contextlib.contextmanager
def instrument_trainer(trainer, spec, probe_protocol, *, probe_class=FixedProbe):
    """Scoped observers; not a replacement optimizer or a modified sampler."""
    names = ("make_train_eval_datasets", "make_pre_post_processors", "update_policy")
    original = {key: getattr(trainer, key) for key in names}
    signature = inspect.signature(original["update_policy"])
    state = {"step": 0, "probe": None}

    def datasets(cfg, *args, **kwargs):
        if cfg.resume or cfg.dataset.eval_split != 0 or cfg.dataset.image_transforms.enable:
            raise ValueError("Study requires fresh training without augmentation or another data split")
        if cfg.prompt_rank.freq != 0 or cfg.env_eval_freq != 0 or cfg.eval_steps != 0:
            raise ValueError("Study owns its diagnostic schedule; disable other online probes")
        if cfg.policy.type != "pi05" or not cfg.policy.train_action_expert or cfg.policy.compile_model:
            raise ValueError("Study requires native, uncompiled PI05 prompt + full expert")
        result = original["make_train_eval_datasets"](cfg, *args, **kwargs)
        state.update(dataset=result[0], cfg=cfg)
        return result

    def processors(*args, **kwargs):
        result = original["make_pre_post_processors"](*args, **kwargs)
        state["preprocessor"] = result[0]
        return result

    def update(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        accelerator, optimizer = values["accelerator"], values["optimizer"]
        policy = accelerator.unwrap_model(values["policy"], keep_fp32_wrapper=True)
        if state["probe"] is None:
            state["probe"] = collective(accelerator, lambda: probe_class(
                state["dataset"], state["preprocessor"], policy, probe_protocol,
                spec["run"], accelerator.device))
            state["probe"].save(accelerator)
            collective(accelerator, lambda: write_json(Path(spec["run"]) / "study_spec.json", spec,
                       exclusive=True) if accelerator.is_main_process else None)
            state["probe"].measure(policy, 0, accelerator)
        before = [float(group["lr"]) for group in optimizer.param_groups]
        expected = lr_value(state["step"], steps=spec["steps"], warmup=spec["warmup_steps"],
                            peak=spec["peak_lr"], end=spec["end_lr"])
        if not np.allclose(before, expected, rtol=1e-7, atol=1e-14):
            raise ValueError(f"Unexpected actual LR before update {state['step']}: {before} != {expected}")
        result = original["update_policy"](*args, **kwargs)
        state["step"] += 1
        step = state["step"]
        skipped = bool((result[1] or {}).get("optimizer_step/skipped", 0)) or bool(
            getattr(optimizer, "step_was_skipped", False))
        row = {"step": step, "used_lr": before,
               "next_lr": [float(group["lr"]) for group in optimizer.param_groups],
               "optimizer_step_skipped": skipped}
        def record_lr():
            if accelerator.is_main_process:
                with (Path(spec["run"]) / "curriculum_probe/lr.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        collective(accelerator, record_lr)
        if skipped:
            raise RuntimeError("Optimizer update skipped; stopping rather than claiming a matched budget")
        if step % spec["probe_frequency"] == 0 or step in spec["probe_steps"] or step == spec["steps"]:
            state["probe"].measure(policy, step, accelerator)
        return result

    trainer.make_train_eval_datasets = datasets
    trainer.make_pre_post_processors = processors
    trainer.update_policy = update
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(trainer, key, value)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec, probe = check_spec(args.spec)
    if Path(spec["run"]).exists():
        raise FileExistsError(f"Output exists; no implicit overwrite/resume: {spec['run']}")
    from lerobot.scripts import lerobot_train as trainer
    previous = sys.argv
    try:
        sys.argv = ["lerobot-train", *spec["training_args"]]
        with instrument_trainer(trainer, spec, probe):
            trainer.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
