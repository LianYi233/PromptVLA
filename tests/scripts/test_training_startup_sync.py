"""Exercise the real training entry point with delayed ranks, without loading a VLA."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from accelerate.utils import DistributedType


def load_entry_point(namespace):
    # Execute the actual entry-point control flow. Exclude module-level optional
    # robot/model imports and the CLI parser; neither is relevant to this race.
    path = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    function = next(
        node for node in ast.parse(path.read_text()).body
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

    train = load_entry_point({
        "_make_training_accelerator": make_accelerator,
        "_pi05_next_action_pretraining_active": lambda cfg: False,
        "_train_single_stage": train_stage,
    })
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
