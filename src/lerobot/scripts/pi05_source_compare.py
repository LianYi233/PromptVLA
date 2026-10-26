"""Compare original PI05 weight sources with identical LIBERO processing and P/D/E/T inputs.

Preparation is CPU-only: copy small configuration files and symlink original weights.
No checkpoint tensor is mixed, updated or randomly initialized by this helper.
"""

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open

from lerobot.scripts.libero_vlm_only import assert_same_processors, processor_files, validate_source
from lerobot.utils.prompt_init_names import resolve_libero_prompt
from lerobot.utils.prompt_initialization import read_initial_prompt

PROTOCOL = "pi05-weight-source-pair-v1"
LABELS = ("libero", "generic")
NAME = re.compile(r"pi05-source-r([1-9][0-9]*)-(libero|generic)-W0-Bocc-base-vlm2-P1-D0-E1-T1")
ARCH = ("paligemma_variant", "action_expert_variant", "max_action_dim", "max_state_dim")
ROLE = "model.inpainting_visible_action_embedding.weight"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def dataset_inventory(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
        for folder in ("data", "videos", "meta")
        for p in sorted((root / folder).rglob("*"))
        if p.is_file()
    }


def weight_schema(path):
    """Header-only compatibility check, mirroring supported PI05 loader aliases.

    Runtime loading remains strict. Dtypes may differ: both runs explicitly use bf16.
    Reject learned prompts and nonzero inpainting additions in a purported base model.
    """
    schema = {}
    with safe_open(path, framework="np") as weights:
        for key in weights.keys():  # noqa: SIM118 - safe_open is not an iterable mapping.
            shape = weights.get_slice(key).get_shape()
            if key.endswith(("vlm_prompt_tokens.weight", "prompt_tokens.weight")):
                if not shape or shape[0] != 0:
                    raise ValueError(f"Source contains learned prompts: {key}")
                # The loader cannot replace an explicit zero-length VLM tensor with VLM2.
                raise ValueError("Use original official weights without stored prompt tensors")
            if key == ROLE:
                if weights.get_tensor(key).any():
                    raise ValueError("Source contains a nonzero inpainting role embedding")
                continue
            if key.startswith("state_proj.") or re.fullmatch(
                r"paligemma_with_expert\.gemma_expert\.model\.(layers\.\d+\."
                r"(input_layernorm|post_attention_layernorm)|norm)\.weight",
                key,
            ):
                continue
            name = key
            if key.startswith("action_time_mlp_"):
                name = key.replace("action_time_mlp_", "time_mlp_", 1)
            if not name.startswith("model."):
                name = "model." + name
            if name in schema:
                raise ValueError(f"Duplicate remapped weight: {name}")
            schema[name] = shape
        head = "model.paligemma_with_expert.paligemma.lm_head.weight"
        if head in schema:
            schema["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = schema[
                head
            ]
    if not schema:
        raise ValueError("Empty weight checkpoint")
    return schema


def check_name(run, factors, tokens, within_seed):
    match = NAME.fullmatch(Path(run).name)
    if not match or (
        tokens,
        within_seed,
        factors["prompt_seed"],
        factors["data_seed"],
        factors["noise_seed"],
        factors["time_seed"],
        factors["flow_binding"],
    ) != (2, 0, 1, 0, 1, 1, "occurrence"):
        raise ValueError(f"Expected source-comparison P1/D0/W0/E1/T1/Bocc name/config: {run}")
    return int(match[1])


def prepare(libero, generic, plan_path, prompt_path, dataset_root, reference_run, output):
    from lerobot.utils.pdet_order import arrays_hash, load_plan, read_frame_metadata

    paths = [libero, generic, plan_path, prompt_path, dataset_root, reference_run, output]
    libero, generic, plan_path, prompt_path, dataset_root, reference_run, output = [
        Path(p).resolve() for p in paths
    ]
    if output.exists():
        raise FileExistsError(
            f"Prepared comparison already exists: {output}; use check, or a new SOURCE_PAIR_DIR"
        )
    validate_source(libero, "base", dataset_root)
    configs = {
        label: read_json(path / "config.json") for label, path in zip(LABELS, (libero, generic), strict=True)
    }
    for key in ARCH:
        if key not in configs["libero"] or configs["generic"].get(key) != configs["libero"][key]:
            raise ValueError(f"Different/missing backbone architecture field: {key}")
    if configs["generic"].get("type") != "pi05":
        raise ValueError("pi05_base must be a LeRobot PI05 safetensors checkpoint, not OpenPI/JAX")
    if (configs["libero"].get("chunk_size"), configs["libero"].get("max_action_dim")) != (50, 32):
        raise ValueError("Expected original chunk50/max_action_dim32 recipe")
    if configs["libero"].get("input_features", {}).get("observation.state", {}).get("shape") != [8] or (
        configs["libero"].get("output_features", {}).get("action", {}).get("shape") != [7]
    ):
        raise ValueError("Reference must use LIBERO state8/action7 features")
    schemas = {
        label: weight_schema(str(path / "model.safetensors"))
        for label, path in zip(LABELS, (libero, generic), strict=True)
    }
    if schemas["libero"] != schemas["generic"]:
        changed = [
            key
            for key in schemas["libero"].keys() | schemas["generic"].keys()
            if schemas["libero"].get(key) != schemas["generic"].get(key)
        ]
        raise ValueError(
            f"Checkpoint key/shape mismatch (not a weights-only comparison): {sorted(changed)[:12]}"
        )
    plan = load_plan(plan_path)
    info = plan["manifest"]
    if tuple(info[k] for k in ("data_seed", "within_seed", "steps", "batch_size", "world_size")) != (
        0,
        0,
        3000,
        32,
        2,
    ):
        raise ValueError("Expected existing D0/W0 plan, 3000 steps, 2 GPUs x batch32")
    if arrays_hash(read_frame_metadata(dataset_root)) != info["dataset_metadata_sha256"]:
        raise ValueError("Dataset differs from the original W0 plan")
    _, prompt = read_initial_prompt(prompt_path, 2)
    if prompt["source_training_seed"] != 1 or prompt.get("source_prompt_init_path"):
        raise ValueError("P1 must be the saved original step-zero seed1 prompt")
    ref_path = reference_run / "pdet/manifest-rank0.json"
    reference = read_json(ref_path)
    if reference["order_plan"] != info or reference["prompt_sha256"] != prompt["prompt_sha256"]:
        raise ValueError("Plan or exact prompt tensor differs from reference run")
    if Path(reference["source_pretrained_path"]).resolve() != libero:
        raise ValueError("LIBERO source is not the reference run's original source")
    f = reference["factors"]
    if tuple(f[k] for k in ("prompt_seed", "data_seed", "noise_seed", "time_seed", "flow_binding")) != (
        1,
        0,
        1,
        1,
        "occurrence",
    ) or (reference["global_seed"], reference["tokens"], reference["total_steps"]) != (0, 2, 3000):
        raise ValueError("Reference must be P1/D0/E1/T1/Bocc, VLM2, global seed0, 3000 steps")
    sources = {}
    for label, source in zip(LABELS, (libero, generic), strict=True):
        print(f"Hashing original {label} weights once on CPU: {source}", flush=True)
        sources[label] = {
            "weights": {
                **identity(source / "model.safetensors"),
                "sha256": digest(source / "model.safetensors"),
            },
            "original_config": configs[label],
            "original_config_sha256": digest(source / "config.json"),
        }
    if sources["libero"]["weights"]["sha256"] == sources["generic"]["weights"]["sha256"]:
        raise ValueError("Both sources contain identical weight files; this is not a source comparison")
    small_files = ["config.json", *processor_files(libero)]
    record = {
        "protocol": PROTOCOL,
        "sources": sources,
        "config_source": str(libero),
        "shared_files": {name: digest(libero / name) for name in small_files},
        "dataset_root": str(dataset_root),
        "dataset_inventory": dataset_inventory(dataset_root),
        "dataset_stats_sha256": digest(dataset_root / "meta/stats.json"),
        "reference_run": str(reference_run),
        "reference_manifest_sha256": digest(ref_path),
        "plan_sha256": digest(plan_path),
        "order_plan": info,
        "prompt_file_sha256": digest(prompt_path),
        "prompt_sha256": prompt["prompt_sha256"],
        "shared_prompt_provenance": {
            "filename": prompt_path.name,
            "source_pretrained_path": prompt.get("source_pretrained_path"),
            "source_training_seed": prompt["source_training_seed"],
            "scope": "The SAME saved P1 tensor is used for both weight sources.",
        },
        "weight_schema_sha256": hashlib.sha256(
            json.dumps(schemas["libero"], sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Only weight source differs; shared LIBERO config/processors and original P1/D0/W0/E1/T1. No numerical determinism guarantee.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-pair-", dir=output.parent) as tmp:
        temp = Path(tmp)
        for label in LABELS:
            view = temp / label
            view.mkdir()
            for name in small_files:
                (view / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(libero / name, view / name)
            (view / "model.safetensors").symlink_to(sources[label]["weights"]["path"])
        shutil.copyfile(plan_path, temp / "D0-W0.npz")
        shutil.copyfile(prompt_path, temp / "P1.safetensors")
        (temp / "pair.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    return {
        "prepared": "passed",
        "directory": str(output),
        "sources": sources,
        "note": "Two views share LIBERO processing; symlinks point to distinct original weight files.",
    }


def check(directory, *, full_hash=False):
    directory = Path(directory).resolve()
    record = read_json(directory / "pair.json")
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Unknown source comparison protocol")
    for label in LABELS:
        view = directory / label
        weights = record["sources"][label]["weights"]
        if identity(view / "model.safetensors") != {k: weights[k] for k in ("path", "size", "mtime_ns")}:
            raise ValueError(f"Original weight file/link changed: {label}")
        if full_hash and digest(view / "model.safetensors") != weights["sha256"]:
            raise ValueError(f"Original weight contents changed: {label}")
        for filename, expected in record["shared_files"].items():
            if digest(view / filename) != expected:
                raise ValueError(f"Shared config/processor changed: {label}/{filename}")
    for filename, key in (("D0-W0.npz", "plan_sha256"), ("P1.safetensors", "prompt_file_sha256")):
        if digest(directory / filename) != record[key]:
            raise ValueError(f"Shared training input changed: {filename}")
    if dataset_inventory(record["dataset_root"]) != record["dataset_inventory"] or (
        digest(Path(record["dataset_root"]) / "meta/stats.json") != record["dataset_stats_sha256"]
    ):
        raise ValueError("Dataset inventory or normalization statistics changed after preparation")
    return record


def check_checkpoint(directory, run, *, step=3000):
    from lerobot.scripts.libero_vlm_only import validate_checkpoint

    directory, run = Path(directory).resolve(), Path(run)
    check(directory)
    config_path = run / "checkpoints" / f"{step:06d}" / "pretrained_model"
    config = read_json(config_path / "train_config.json")
    check_name(run, config["pdet"], config["policy"]["num_vlm_prompt_tokens"], 0)
    label = NAME.fullmatch(run.name)[2]
    if config["seed"] != 0 or config["steps"] != 3000 or step != 3000:
        raise ValueError("Expected global seed0 and final checkpoint3000")
    validate_checkpoint(config_path, 2, source=directory / label, kind="base")
    return {"checkpoint": "passed", "run": str(run), "weight_source": label}


def audit(directory, runs):
    from lerobot.scripts.pdet_order_binding import inspect_run

    directory = Path(directory).resolve()
    record = check(directory)
    if len(runs) != 2:
        raise ValueError("Audit needs the two completed libero/generic runs")
    shared, reports, labels = [], [], set()
    processors = []
    for run in runs:
        invariant, report = inspect_run(run, name_checker=check_name)
        label = NAME.fullmatch(Path(run).name)[2]
        labels.add(label)
        expected_source = directory / label
        if Path(invariant.pop("source_pretrained_path")).resolve() != expected_source:
            raise ValueError("Run loaded an unexpected source")
        recipe = copy.deepcopy(invariant["recipe"])
        if Path(recipe["policy"].pop("pretrained_path")).resolve() != expected_source:
            raise ValueError("Recipe loaded an unexpected source")
        invariant["recipe"] = recipe
        if invariant["prompt_sha256"] != record["prompt_sha256"]:
            raise ValueError("Initial prompt differs from the saved P1 tensor")
        if Path(recipe["dataset"]["root"]).resolve() != Path(record["dataset_root"]):
            raise ValueError("Dataset path differs from preparation")
        check_checkpoint(directory, run)
        processors.append(Path(run) / "checkpoints/003000/pretrained_model")
        shared.append(invariant)
        reports.append(
            {
                **report,
                "weight_source": label,
                "weights_sha256": record["sources"][label]["weights"]["sha256"],
            }
        )
    if labels != set(LABELS) or shared[0] != shared[1]:
        raise ValueError(
            "Expected both sources with identical remaining recipe, initialization, data and code"
        )
    if any(reports[0][key] != reports[1][key] for key in ("bank_sha256", "order_sha256")):
        raise ValueError("Realized sample/flow inputs differ between weight sources")
    assert_same_processors(*processors)
    # The historical run can use another naming/code protocol; compare inputs only.
    reference = Path(record["reference_run"])
    if digest(reference / "pdet/manifest-rank0.json") != record["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    ref_inv, ref_report = inspect_run(reference, name_checker=lambda *_: 0)
    for key in ("prompt_sha256", "pool", "tasks", "shapes", "global_seed", "dataset_metadata_sha256"):
        if ref_inv[key] != shared[0][key]:
            raise ValueError(f"Input differs from reference: {key}")
    for key in ("bank_sha256", "order_sha256"):
        if ref_report[key] != reports[0][key]:
            raise ValueError(f"Realized input differs from reference: {key}")
    return {
        "audit": "passed",
        "runs": reports,
        "reference_inputs": "matched",
        "scope": "Full sample/flow traces, exact P1 and matched recipe/processors across new sources; historical inputs only, not historical code or numerical equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--libero", type=Path, default=Path("/home/wyn/models/lerobot/pi05_libero_base"))
    prep.add_argument("--generic", type=Path, default=Path("/home/wyn/models/lerobot/pi05_base"))
    prep.add_argument("--old-plan", type=Path, default=Path("/data1/wyn/prompt-work/keyframe-r1/D0-W0.npz"))
    prep.add_argument(
        "--prompt", type=Path, help="Explicit saved P1; default prefers the source-named LIBERO initializer"
    )
    prep.add_argument("--prompt-init-dir", type=Path, default=Path("/data1/wyn/prompt-work"))
    prep.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    prep.add_argument(
        "--reference-run",
        type=Path,
        default=Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-key-r1-B-W0-Bocc-base-vlm2-P1-D0-E1-T1"),
    )
    prep.add_argument("--output", type=Path, default=Path("/data1/wyn/prompt-work/source-pair-r1"))
    verify = sub.add_parser("check")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--full-hash", action="store_true")
    audit_cmd = sub.add_parser("audit")
    audit_cmd.add_argument("directory", type=Path)
    audit_cmd.add_argument("runs", type=Path, nargs="+")
    checkpoint = sub.add_parser("check-checkpoint")
    checkpoint.add_argument("directory", type=Path)
    checkpoint.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.libero,
            args.generic,
            args.old_plan,
            args.prompt or resolve_libero_prompt(args.prompt_init_dir, 1),
            args.dataset_root,
            args.reference_run,
            args.output,
        )
    elif args.command == "check":
        check(args.directory, full_hash=args.full_hash)
        result = {"preflight": "passed", "directory": str(args.directory)}
    elif args.command == "audit":
        result = audit(args.directory, args.runs)
    else:
        result = check_checkpoint(args.directory, args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
