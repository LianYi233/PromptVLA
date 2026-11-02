"""Exercise the real training entry point with delayed ranks, without loading a VLA."""

import ast
import json
import logging
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType


def load_entry_point(namespace):
    # Execute the actual entry-point control flow. Exclude module-level optional
    # robot/model imports and the CLI parser; neither is relevant to this race.
    path = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    namespace["TrainPipelineConfig"] = object
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["train"]


class Config:
    def __init__(self, output, rank, accelerator_ready, first_write, *, delay):
        self.output_dir = output
        self.rank = rank
        self.accelerator_ready = accelerator_ready
        self.first_write = first_write
        self.delay = delay
        self.job = SimpleNamespace(is_remote=False)
        self.prompt_rank = SimpleNamespace(freq=0)
        self.pdet = SimpleNamespace(enabled=False)
        self.piper_eval = SimpleNamespace(enabled=False)
        self.vlm_prompt_init_path = self.vlm_prompt_export_path = None
        self.cabo_active = False
        self.validated = False

    def validate(self):
        if self.delay and self.rank == 1:
            assert self.accelerator_ready.wait(timeout=5)
            # Give rank 0 a deterministic opportunity to create the directory.
            # The fixed entry point blocks it at a barrier until we validate.
            self.first_write.wait(timeout=0.3)
        if self.output_dir.exists():
            raise FileExistsError(f"Output directory {self.output_dir} already exists")
        self.validated = True


class Accelerator:
    distributed_type = DistributedType.MULTI_CPU

    def __init__(self, barrier):
        self.barrier = barrier
        self.ended = False

    def wait_for_everyone(self):
        self.barrier.wait(timeout=5)

    def end_training(self):
        self.ended = True


@pytest.mark.parametrize("existing", [False, True])
def test_startup_waits_for_slow_rank_and_preserves_existing_directory_guard(tmp_path, existing):
    output = tmp_path / "run"
    if existing:
        output.mkdir()
        (output / "keep.txt").write_text("existing run")
    ready, first_write = Event(), Event()
    configs = [Config(output, rank, ready, first_write, delay=not existing) for rank in (0, 1)]
    barrier = Barrier(2)
    accelerators = [Accelerator(barrier) for _ in configs]

    def make_accelerator(cfg, accelerator):
        if cfg.rank == 0:
            ready.set()
        return accelerator

    def train_stage(cfg, accelerator):
        if cfg.rank == 0:
            # Same first filesystem side effect as WandBLogger.
            output.mkdir(parents=True, exist_ok=True)
            first_write.set()
        assert all(other.validated for other in configs)

    train = load_entry_point(
        {
            "_make_training_accelerator": make_accelerator,
            "_pi05_next_action_pretraining_active": lambda cfg: False,
            "_train_single_stage": train_stage,
        }
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train, cfg, acc) for cfg, acc in zip(configs, accelerators, strict=True)]
        if existing:
            for future in futures:
                with pytest.raises(FileExistsError, match="already exists"):
                    future.result(timeout=10)
        else:
            for future in futures:
                future.result(timeout=10)
    if existing:
        assert not first_write.is_set()
        assert (output / "keep.txt").read_text() == "existing run"
    else:
        assert output.is_dir()
        assert all(acc.ended for acc in accelerators)


def load_wandb_startup(namespace):
    """Load the production entry point and startup path, excluding model imports."""
    path = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    names = {"train", "_train_single_stage", "_initialize_wandb_logger"}
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for function in functions:
        function.decorator_list = []
    namespace.update(
        TrainPipelineConfig=object,
        torch=torch,
        logging=logging,
        colored=lambda text, *args, **kwargs: text,
        pformat=repr,
    )
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def wandb_config(**overrides):
    wandb = {"enable": True, "project": "startup-test", "mode": "online"}
    wandb.update(overrides)
    return SimpleNamespace(wandb=SimpleNamespace(**wandb), vlm_prompt_export_path=None)


class SimulatedWandB403Error(Exception):
    pass


class StartupReachedSeedError(Exception):
    """End the actual startup path before dataset/model allocation in tests."""


def test_single_process_wandb_keeps_original_exception_and_online_mode():
    original = SimulatedWandB403Error("Error uploading run: returned error 403 Forbidden")

    def factory(cfg):
        raise original

    namespace = load_wandb_startup({"WandBLogger": factory})
    accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
    cfg = wandb_config()
    with pytest.raises(SimulatedWandB403Error) as caught:
        namespace["_initialize_wandb_logger"](cfg, accelerator)
    assert caught.value is original
    assert cfg.wandb.mode == "online"


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("reason", ("disabled_flag", "disabled_mode", "no_project", "prompt_export"))
def test_disabled_wandb_and_prompt_exports_never_call_sdk(rank, reason):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled W&B must not call its SDK or a logging collective")

    namespace = load_wandb_startup({"WandBLogger": forbidden})
    accelerator = SimpleNamespace(is_main_process=rank == 0, num_processes=2, reduce=forbidden)
    cfg = wandb_config()
    if reason == "disabled_flag":
        cfg.wandb.enable = False
    elif reason == "disabled_mode":
        cfg.wandb.mode = "disabled"
    elif reason == "no_project":
        cfg.wandb.project = ""
    else:
        cfg.vlm_prompt_export_path = "prompt.safetensors"
    assert namespace["_initialize_wandb_logger"](cfg, accelerator) is None


def _wandb_gloo_worker(rank, rendezvous, result_dir, scenario):
    """Two real Gloo ranks execute train(), including Accelerator.end_training()."""
    import torch.distributed as dist
    from accelerate import Accelerator as RealAccelerator

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )
    torch.set_num_threads(1)
    record = {"rank": rank, "sdk_calls": 0, "reductions": 0, "barriers": 0, "cleanup_calls": 0}
    original = SimulatedWandB403Error("Error uploading run: returned error 403 Forbidden")
    try:
        dist.init_process_group(
            "gloo",
            init_method=Path(rendezvous).as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=10),
        )
        accelerator = RealAccelerator(cpu=True, mixed_precision="no")
        assert accelerator.num_processes == 2
        actual_reduce, actual_barrier, actual_end = (
            accelerator.reduce,
            accelerator.wait_for_everyone,
            accelerator.end_training,
        )

        def reduce(*args, **kwargs):
            record["reductions"] += 1
            return actual_reduce(*args, **kwargs)

        def barrier():
            record["barriers"] += 1
            return actual_barrier()

        def end():
            record["cleanup_calls"] += 1
            return actual_end()

        accelerator.reduce, accelerator.wait_for_everyone, accelerator.end_training = reduce, barrier, end

        def factory(cfg):
            record["sdk_calls"] += 1
            assert rank == 0, "W&B must only initialize on global main"
            if scenario == "failure":
                # Let the peer enter the startup reduction first, reproducing the
                # ordering that used to leave it stuck at the dataset barrier.
                time.sleep(0.1)
                raise original
            return SimpleNamespace()

        def seed(*args, **kwargs):
            raise StartupReachedSeedError()

        namespace = load_wandb_startup(
            {
                "WandBLogger": factory,
                "_make_training_accelerator": lambda cfg, accelerator: accelerator,
                "_pi05_next_action_pretraining_active": lambda cfg: False,
                "init_logging": lambda **kwargs: None,
                "set_seed": seed,
            }
        )
        cfg = wandb_config(mode="disabled" if scenario == "disabled" else "online")
        cfg.job = SimpleNamespace(is_remote=False)
        cfg.prompt_rank = SimpleNamespace(freq=0)
        cfg.pdet = SimpleNamespace(enabled=False)
        cfg.piper_eval = SimpleNamespace(enabled=False)
        cfg.vlm_prompt_init_path = None
        cfg.cabo_active = False
        cfg.trainable_config = SimpleNamespace()
        cfg.seed = 0
        cfg.validate = lambda: None
        cfg.to_dict = lambda: {}
        start = time.monotonic()
        try:
            namespace["train"](cfg, accelerator)
        except StartupReachedSeedError:
            record["outcome"] = "ready"
        except Exception as error:
            record.update(
                outcome="error",
                error_type=type(error).__name__,
                error=str(error),
                original_exception=error is original,
                traceback="".join(traceback.format_exception(error)),
            )
        record["elapsed"] = time.monotonic() - start
        record["group_destroyed"] = not dist.is_initialized()
        record["mode"] = cfg.wandb.mode
    except Exception:
        record["unexpected"] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(record))


@pytest.mark.parametrize("scenario", ("failure", "success", "disabled"))
def test_wandb_startup_real_gloo_ranks_exit_together_and_cleanup(tmp_path, scenario):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch Gloo backend is unavailable")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_wandb_gloo_worker,
            args=(rank, str(tmp_path / "rendezvous"), str(tmp_path), scenario),
        )
        for rank in (0, 1)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert not any(process.is_alive() for process in processes), "W&B failure stranded a rank"
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    for record in records:
        assert "unexpected" not in record, record.get("unexpected")
        assert record["group_destroyed"] and record["cleanup_calls"] == 1
        assert record["elapsed"] < 6, record
        assert record["barriers"] == 1  # The entry barrier; no peer reaches a later data barrier.
        assert record["reductions"] == (0 if scenario == "disabled" else 1)
        assert record["sdk_calls"] == (1 if scenario != "disabled" and record["rank"] == 0 else 0)
        assert record["mode"] == ("disabled" if scenario == "disabled" else "online")
    if scenario == "failure":
        assert records[0]["error_type"] == "SimulatedWandB403Error"
        assert records[0]["original_exception"]
        assert "403 Forbidden" in records[0]["error"]
        assert "factory" in records[0]["traceback"]  # Preserve the actual SDK failure frame.
        assert records[1]["error_type"] == "RuntimeError"
        assert "W&B initialization failed on the main process" in records[1]["error"]
        assert all(record["outcome"] == "error" for record in records)
    else:
        assert all(record["outcome"] == "ready" for record in records)
