"""Audit the full order x binding experiment, using the SAME indexed noise/time bank."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from lerobot.scripts.pdet_followup import read_run
from lerobot.utils.pdet_order import FLOW_PROTOCOL, META_COLUMNS, SLOT_FLOW_PROTOCOL, arrays_hash, load_plan

NAME = re.compile(
    r"pi05-orderbind-r([1-9][0-9]*)-W([0-9]+)-B(occ|slot)-base-vlm([0-9]+)"
    r"-P([0-9]+)-D([0-9]+)-E([0-9]+)-T([0-9]+)"
)
SHARED = (
    "global_seed", "world_size", "per_device_batch_size", "tokens", "dtype", "prompt_sha256",
    "source_pretrained_path", "dataset_root", "dataset_metadata_sha256", "versions", "cuda",
    "recipe", "git_commit", "source_tree_sha256", "total_steps", "flow_key_namespace",
)


def check_name(run, factors, tokens, within_seed):
    match = NAME.fullmatch(Path(run).name)
    binding = {"occurrence": "occ", "slot": "slot"}.get(factors.get("flow_binding"))
    if match is None or binding is None:
        raise ValueError(f"Not an order-binding run name/config: {run}")
    experiment, w, label, n, p, d, e, t = match.groups()
    expected = (within_seed, tokens, *(factors[key] for key in (
        "prompt_seed", "data_seed", "noise_seed", "time_seed"
    )))
    if tuple(map(int, (w, n, p, d, e, t))) != expected or label != binding:
        raise ValueError(f"Run name does not match W/binding/PDET/token config: {run}")
    return int(experiment)


def check_checkpoint(run, step=3000):
    """Evaluation preflight, before the existing evaluator loads expensive weights."""
    path = Path(run) / "checkpoints" / f"{step:06d}" / "pretrained_model/train_config.json"
    config = json.loads(path.read_text())
    factors = config["pdet"]
    plan = load_plan(factors["order_plan"])
    if not factors["enabled"] or config["seed"] != 0 or config["steps"] != step:
        raise ValueError("Unexpected checkpoint training configuration")
    check_name(run, factors, config["policy"]["num_vlm_prompt_tokens"], plan["manifest"]["within_seed"])
    return {"checkpoint": "name_and_config_match", "run": str(run)}


def inspect_run(run, *, name_checker=check_name, bank_prefix_count=None):
    run = Path(run)
    manifest, rows = read_run(run)
    factors = manifest["factors"]
    binding = factors.get("flow_binding")
    expected_protocol = {"occurrence": FLOW_PROTOCOL, "slot": SLOT_FLOW_PROTOCOL}.get(binding)
    if expected_protocol is None or manifest["protocol"] != expected_protocol:
        raise ValueError(f"Unexpected flow binding/protocol: {run}")
    if manifest.get("flow_key_namespace") != FLOW_PROTOCOL:
        raise ValueError("Both bindings must use the same indexed CPU generator namespace")
    if factors["audit_steps"] < manifest["total_steps"]:
        raise ValueError("Order-binding audit requires the complete realized training trace")
    if not manifest.get("git_commit") or not manifest.get("source_tree_sha256"):
        raise ValueError("Missing training code provenance")
    for path in (run / "pdet").glob("manifest-rank*.json"):
        other = json.loads(path.read_text())
        if any(other.get(key) != manifest.get(key) for key in SHARED):
            raise ValueError(f"Different configuration/source across ranks: {run}")
    plan = load_plan(factors["order_plan"])
    info = plan["manifest"]
    if info != manifest["order_plan"] or info["data_seed"] != factors["data_seed"]:
        raise ValueError("Recorded plan was changed or D differs")
    if (info["steps"], info["batch_size"], info["world_size"]) != (
        manifest["total_steps"], manifest["per_device_batch_size"], manifest["world_size"]
    ):
        raise ValueError("Plan and runtime steps/batch/world size differ")
    if info["dataset_metadata_sha256"] != manifest["dataset_metadata_sha256"]:
        raise ValueError("Recorded dataset identity differs from plan")
    experiment = name_checker(run, factors, manifest["tokens"], info["within_seed"])
    batch = info["batch_size"]
    count = len(plan["order"])
    bank_rows = [None] * count
    shapes = None
    for row in rows:
        start = (row["step"] * info["world_size"] + row["rank"]) * batch
        slots = list(range(start, start + batch))
        ids = row["occurrence_ids"]
        if ids != plan["order"][start : start + batch].tolist():
            raise ValueError("Realized sample order differs from the planned DDP slots")
        for key in META_COLUMNS:
            recorded = row["sample_ids"] if key == "index" else row[key]
            if recorded != plan[key][ids].tolist():
                raise ValueError(f"Realized {key} differs from the fixed sample pool")
        keys = ids if binding == "occurrence" else slots
        if row.get("flow_key_ids") != keys:
            raise ValueError("Flow key does not follow the declared binding")
        current_shapes = (row["noise_shape"], row["time_shape"])
        if shapes is None:
            shapes = current_shapes
        if shapes != current_shapes or shapes[0][0] != batch or shapes[1] != [batch]:
            raise ValueError("Noise/time shapes changed within the experiment")
        for key in ("noise", "time"):
            if len(row[f"occurrence_{key}_sha256"]) != batch:
                raise ValueError("Missing per-sample flow hashes")
        for i, key in enumerate(keys):
            if type(key) is not int or not 0 <= key < count or bank_rows[key] is not None:
                raise ValueError("Repeated or invalid indexed-bank key")
            hashes = [row[f"occurrence_{factor}_sha256"][i] for factor in ("noise", "time")]
            if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
                raise ValueError("Invalid flow hash")
            bank_rows[key] = bytes.fromhex("".join(hashes))
    if any(row is None for row in bank_rows):
        raise ValueError("Missing indexed-bank keys in the realized trace")
    digest = hashlib.sha256()
    for row in bank_rows:
        digest.update(row)
    invariant = {key: manifest[key] for key in SHARED}
    invariant["factors"] = {
        key: value for key, value in factors.items()
        if key not in ("order_plan", "flow_binding", "noise_seed", "time_seed")
    }
    invariant.update(
        experiment=experiment,
        pool=info["instance_pool_sha256"],
        tasks=info["task_schedule_sha256"],
        shapes=shapes,
    )
    report = {
        "run": str(run), "within_seed": info["within_seed"], "binding": binding,
        "noise_seed": factors["noise_seed"], "time_seed": factors["time_seed"],
        "occurrences": count, "bank_sha256": digest.hexdigest(),
        "order_sha256": arrays_hash({"order": np.asarray(plan["order"])}),
    }
    if bank_prefix_count is not None:
        if type(bank_prefix_count) is not int or not 1 <= bank_prefix_count <= count:
            raise ValueError("Invalid bank prefix count")
        prefix = hashlib.sha256()
        for row in bank_rows[:bank_prefix_count]:
            prefix.update(row)
        report.update(bank_prefix_count=bank_prefix_count, bank_prefix_sha256=prefix.hexdigest())
    return invariant, report


def audit_runs(runs, *, allow_partial=False):
    reference = None
    reports, orders, banks, cells = [], {}, {}, set()
    for run in runs:
        invariant, report = inspect_run(run)
        if reference is None:
            reference = invariant
        elif invariant != reference:
            raise ValueError("Changed recipe/source/P/D/sample pool/task slots across runs")
        w, binding = report["within_seed"], report["binding"]
        bank = report["noise_seed"], report["time_seed"]
        cell = (*bank, w, binding)
        if cell in cells:
            raise ValueError(f"Duplicate experimental cell: {cell}")
        cells.add(cell)
        if w in orders and orders[w] != report["order_sha256"]:
            raise ValueError("W label has different sample orders across bindings or noise banks")
        orders[w] = report["order_sha256"]
        if bank in banks and banks[bank] != report["bank_sha256"]:
            raise ValueError("The realized noise/time BANK differs across orders or bindings")
        banks[bank] = report["bank_sha256"]
        reports.append(report)
    if not reports:
        raise ValueError("No runs selected")
    if len(set(orders.values())) != len(orders):
        raise ValueError("Different W labels did not change order")
    if len(set(banks.values())) != len(banks):
        raise ValueError("Different E/T banks produced identical recorded draws")
    expected = {(*bank, w, binding) for bank in banks for w in orders for binding in ("occurrence", "slot")}
    complete = len(orders) >= 2 and cells == expected
    if not allow_partial and not complete:
        raise ValueError(f"Incomplete order x binding factorial; missing {sorted(expected - cells)}")
    return {
        "audit": "passed" if complete else "partial_inputs_passed",
        "complete_factorial": complete,
        "noise_banks": len(banks),
        "code_revision": reference["git_commit"],
        "scope": "All steps: same task slots, sample pool, recipe/source and per-key CPU noise/time bank; not numerical determinism",
        "runs": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("runs", type=Path, nargs="+")
    audit.add_argument("--allow-partial", action="store_true")
    check = sub.add_parser("check-checkpoint")
    check.add_argument("run", type=Path)
    check.add_argument("--step", type=int, default=3000)
    args = parser.parse_args()
    result = (
        audit_runs(args.runs, allow_partial=args.allow_partial)
        if args.command == "audit" else check_checkpoint(args.run, args.step)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
