"""Run the checked-out repository's actual plan/scheduler bodies without heavy imports."""
import ast
import copy
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from examples.analysis.curriculum_study import core
from test_protocol import fixture

ROOT = Path(os.environ.get("CURRICULUM_NATIVE_ROOT", Path(__file__).resolve().parents[2]))


def source(path):
    file = ROOT / path
    if not file.exists():
        pytest.skip("Full native checkout is not present; standalone tests still run")
    return ast.parse(file.read_text())


def test_new_plans_load_in_unmodified_native_pdet(tmp_path):
    tree = source("src/lerobot/utils/pdet_order.py")
    names = {"arrays_hash", "load_plan", "validate_plan_dataset"}
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    namespace = {"hashlib": hashlib, "json": json, "np": np, "META_COLUMNS": core.META,
                 "PLAN_PROTOCOL": core.PLAN_PROTOCOL, "FLOW_PROTOCOL": core.FLOW_PROTOCOL}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_pdet", "exec"), namespace)
    base, scores, info = fixture()
    family, _ = core.make_family(base, scores, info, steps=80, warmup_steps=5)
    for order, plan in family.items():
        path = tmp_path / f"{order}.npz"
        core.save_plan(path, plan)
        native = namespace["load_plan"](path)
        namespace["validate_plan_dataset"](native, {k: scores[k] for k in core.META},
                                           steps=80, batch_size=2, world_size=2)


@pytest.mark.parametrize("end", [1e-5, 2.5e-5])
@pytest.mark.parametrize("warmup,decay", [(200, 6000), (1000, 30000), (0, 6000)])
def test_actual_native_scheduler_matches_preupdate_lr_formula(end, warmup, decay):
    tree = source("src/lerobot/optim/schedulers.py")
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and
                             n.name == "CosineDecayWithWarmupSchedulerConfig"))
    # Only strip the plugin registry decorator; retain the actual dataclass and build body.
    node.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"dataclass": dataclass, "Optimizer": torch.optim.Optimizer, "LambdaLR": LambdaLR,
                 "logging": logging, "math": math}
    exec(compile(module, "native_scheduler", "exec"), namespace)
    cls = namespace["CosineDecayWithWarmupSchedulerConfig"]
    p = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([p], lr=2.5e-5)
    scheduler = cls(warmup, decay, 2.5e-5, end).build(optimizer, 6000)
    actual_warmup = int(warmup*6000/decay) if decay > 6000 else warmup
    for step in range(6000):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(core.lr_value(
            step, steps=6000, warmup=actual_warmup, peak=2.5e-5, end=end), rel=1e-12)
        optimizer.step()
        scheduler.step()
