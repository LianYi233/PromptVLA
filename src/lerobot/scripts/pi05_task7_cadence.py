"""Single-demonstration task7 variable-cadence action targets for experiment II.

A fixed finite 1000-update PDET plan controls observation anchors. For each
immutable occurrence, cadence is 1 or 2 *original video frames per action
target*, recorded in an independent manifest. The targets are the original
recorded instantaneous actions at future timestamps, NOT a command to execute
at a new control rate; no robot rollout at 2x cadence is authorized.

All model variants use identical observation anchors, action targets, cadence
labels, and PDET flow draws. Held-out starts are disjoint from training starts;
their future action windows MAY overlap, so they are not independent demos.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.scripts.frameskip_scores import read_actions
from lerobot.scripts import pi05_gripper_event as source
from lerobot.utils.pdet_order import (
    OCCURRENCE_KEY, arrays_hash, load_plan, read_frame_metadata,
    save_plan, validate_plan_dataset,
)

PROTOCOL = "pi05-task7-variable-action-cadence-v1"
DEFAULT_DIR = Path("/data1/wyn/prompt-work/task7-variable-cadence-1k")
SOURCE_DIR = Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base")
HORIZON = 50
TRAIN_STRIDES = (1, 2)
TRAIN_STEPS = 1000
BATCH_SIZE = 32
WORLD_SIZE = 2
DATA_SEED = 0


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def make_action_targets(actions, starts, strides, *, first, last, horizon=HORIZON):
    """Real action[t + j*stride], never reindexing a different episode."""
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    strides = np.asarray(strides, dtype=np.int64).reshape(-1)
    if len(starts) != len(strides):
        raise ValueError("Need one stride for every anchor")
    if not len(starts) or np.any(strides < 1) or np.any(strides > 3):
        raise ValueError("Invalid cadence or empty start list")
    positions = starts[:, None] + np.arange(horizon, dtype=np.int64)[None, :] * strides[:, None]
    if np.any(positions < first) or np.any(positions > last):
        raise ValueError("Cadence target crosses the selected demonstration boundary")
    return np.asarray(actions[positions], dtype=np.float32)


def make_occurrence_strides(count, *, seed=DATA_SEED):
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError("Require positive even number of training occurrences")
    # Equal cardinality exactly. Cadence is assigned to occurrence, not to an
    # index; a repeated source frame can appear with both temporal offsets.
    rng = np.random.default_rng(np.random.SeedSequence([19737, seed]))
    labels = np.tile(np.array(TRAIN_STRIDES, dtype=np.int64), count // 2)
    rng.shuffle(labels)
    return labels


def prepare(
    root=DEFAULT_DIR, *, source_dir=SOURCE_DIR, dataset_root=None,
    heldout_modulo=5, include_stride3=True,
):
    root = Path(root).resolve()
    source_record = source.check(source_dir)
    if (
        source_record["selection"]["task_id"] != 7
        or source_record["settings"]["steps"] != 1000
        or source_record["settings"]["batch_size"] != BATCH_SIZE
        or source_record["settings"]["world_size"] != WORLD_SIZE
    ):
        raise ValueError("Expected verified original one-demo task7 base Uniform plan")
    data_root = Path(dataset_root or source_record["dataset_root"]).resolve()
    if data_root != Path(source_record["dataset_root"]):
        raise ValueError("Dataset root must match the original task7 preparation")
    if type(heldout_modulo) is not int or heldout_modulo < 3:
        raise ValueError("heldout_modulo must be >=3")
    meta = read_frame_metadata(data_root)
    actions = read_actions(data_root)
    task = source.resolve_task(source.read_tasks(data_root), 7)
    ep = int(source_record["selection"]["episode_index"])
    positions = np.flatnonzero(meta["episode_index"] == ep)
    if len(positions) < (HORIZON - 1) * max(TRAIN_STRIDES) + 1:
        raise ValueError("Selected task7 demonstration too short for 50 targets at stride 2")
    if not np.array_equal(positions, np.arange(positions[0], positions[-1]+1)):
        raise ValueError("Episode frames are not contiguous")
    if not np.array_equal(meta["frame_index"][positions], np.arange(len(positions))):
        raise ValueError("Episode frame index is not chronological")
    if not np.all(meta["task_index"][positions] == task["task_index"]):
        raise ValueError("Selected episode is not task7")
    candidates = positions[:len(positions) - (HORIZON - 1)*2]
    if len(candidates) < 10:
        raise ValueError("Not enough same-episode anchors to construct training and pilot holdout")
    heldout = candidates[::heldout_modulo]
    training = candidates[np.arange(len(candidates)) % heldout_modulo != 0]
    if len(training) < 5 or len(heldout) < 2:
        raise ValueError("Not enough training/heldout anchors")
    manifest_selection = {
        **source_record["selection"],
        "source_horizon_max": 99,
        "action_target_horizon": HORIZON,
        "train_cadences": list(TRAIN_STRIDES),
        "heldout_modulo": heldout_modulo,
    }
    plan, _ = source.make_plan(
        meta,
        training,
        np.ones(len(training), dtype=np.float64),
        manifest_selection,
        steps=TRAIN_STEPS, batch_size=BATCH_SIZE, world_size=WORLD_SIZE,
        data_seed=DATA_SEED, sampling="uniform",
    )
    cadence = make_occurrence_strides(len(plan["index"]))
    # All drawn indices share one episode, every future target is in bounds.
    for stride in TRAIN_STRIDES:
        mask = cadence == stride
        positions_for_stride = plan["index"][mask]
        if len(positions_for_stride):
            make_action_targets(actions, positions_for_stride[:1], [stride],
                                first=int(positions[0]), last=int(positions[-1]))
            if np.any(positions_for_stride + (HORIZON-1)*stride > positions[-1]):
                raise ValueError("Training target crosses task7 episode boundary")
    stride3_eval = (
        heldout[heldout + (HORIZON-1)*3 <= positions[-1]].tolist()
        if include_stride3 else []
    )
    record = {
        "protocol": PROTOCOL,
        "dataset_root": str(data_root),
        "source_preparation": str(Path(source_dir).resolve()),
        "source_preparation_sha256": file_hash(Path(source_dir) / "manifest.json"),
        "episode_index": ep,
        "task_id": 7,
        "dataset_task_index": int(task["task_index"]),
        "episode_first": int(positions[0]),
        "episode_last": int(positions[-1]),
        "fps": float(source_record["fps"]),
        "horizon": HORIZON,
        "train_strides": list(TRAIN_STRIDES),
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "world_size": WORLD_SIZE,
        "data_seed": DATA_SEED,
        "actions_sha256": arrays_hash({"actions": actions}),
        "metadata_sha256": arrays_hash(meta),
        "plan_sha256": None,
        "cadence_sha256": None,
        "heldout_starts": heldout.astype(int).tolist(),
        "stride3_heldout_starts": stride3_eval,
        "split_note": "Disjoint STARTS within one demo; future action windows may overlap",
        "target_semantics": "instantaneous recorded action at t+j*stride; NOT aggregated control",
    }
    if root.exists():
        previous = check(root)
        if {k:v for k,v in previous.items() if k not in ("plan_sha256","cadence_sha256")} != {
            k:v for k,v in record.items() if k not in ("plan_sha256","cadence_sha256")
        }:
            raise ValueError("Different variable-cadence preparation already exists")
        return previous
    import tempfile
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task7-cadence-", dir=root.parent) as temp_name:
        temp = Path(temp_name)
        save_plan(temp / "uniform.npz", plan)
        with (temp / "cadence.npz").open("xb") as f:
            np.savez_compressed(f, strides=cadence)
        record["plan_sha256"] = file_hash(temp / "uniform.npz")
        record["cadence_sha256"] = file_hash(temp / "cadence.npz")
        (temp / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        temp.rename(root)
    return record


def check(root=DEFAULT_DIR):
    root = Path(root).resolve()
    rec = load_json(root / "manifest.json")
    if rec.get("protocol") != PROTOCOL:
        raise ValueError("Unknown variable-cadence protocol")
    if file_hash(root / "uniform.npz") != rec["plan_sha256"]:
        raise ValueError("Variable-cadence observation plan changed")
    if file_hash(root / "cadence.npz") != rec["cadence_sha256"]:
        raise ValueError("Variable-cadence per-occurrence stride assignment changed")
    if file_hash(Path(rec["source_preparation"]) / "manifest.json") != rec["source_preparation_sha256"]:
        raise ValueError("Source one-demo task7 preparation changed")
    plan = load_plan(root / "uniform.npz")
    with np.load(root / "cadence.npz", allow_pickle=False) as archive:
        cadence = archive["strides"]
    if (
        cadence.shape != (rec["steps"]*rec["batch_size"]*rec["world_size"],)
        or cadence.dtype != np.int64
        or set(np.unique(cadence)) != {1, 2}
        or abs(int((cadence == 1).sum()) - int((cadence == 2).sum())) != 0
    ):
        raise ValueError("Invalid per-occurrence cadence assignment")
    meta = read_frame_metadata(rec["dataset_root"])
    actions = read_actions(rec["dataset_root"])
    validate_plan_dataset(plan, meta, steps=TRAIN_STEPS, batch_size=BATCH_SIZE, world_size=WORLD_SIZE)
    if arrays_hash({"actions": actions}) != rec["actions_sha256"]:
        raise ValueError("Actions changed since cadence preparation")
    if arrays_hash(meta) != rec["metadata_sha256"]:
        raise ValueError("Task7 metadata changed since cadence preparation")
    if not np.all(meta["episode_index"][plan["index"]] == rec["episode_index"]):
        raise ValueError("Plan includes a different episode")
    if np.any(plan["index"] + (HORIZON-1)*cadence > rec["episode_last"]):
        raise ValueError("Plan includes an out-of-range cadence target")
    if set(rec["heldout_starts"]).intersection(set(plan["index"].tolist())):
        raise ValueError("Train and evaluation starts overlap")
    return rec


class CadencePlannedDataset(torch.utils.data.Dataset):
    """Replace default continuous action labels with exact strided future targets."""

    def __init__(self, dataset, prepared_root, *, validate=True):
        self.dataset = dataset
        self.root = Path(prepared_root).resolve()
        self.manifest = check(self.root) if validate else load_json(self.root / "manifest.json")
        self.plan = load_plan(self.root / "uniform.npz")
        with np.load(self.root / "cadence.npz", allow_pickle=False) as f:
            self.cadences = f["strides"]
        self.actions = read_actions(self.manifest["dataset_root"])
        if Path(self.dataset.root).resolve() != Path(self.manifest["dataset_root"]):
            raise ValueError("Dataset root differs from prepared cadence root")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, occurrence = item
        index, occurrence = int(index), int(occurrence)
        if index != int(self.plan["index"][occurrence]):
            raise ValueError("Occurrence no longer maps to the recorded frame")
        stride = int(self.cadences[occurrence])
        record = dict(self.dataset[index])
        actual_index = record["index"]
        if int(actual_index) != index:
            raise ValueError("Source image and target action frame no longer align")
        actions = make_action_targets(
            self.actions, [index], [stride],
            first=self.manifest["episode_first"], last=self.manifest["episode_last"],
        )[0]
        record["action"] = torch.from_numpy(actions.copy())
        record["action_is_pad"] = torch.zeros(HORIZON, dtype=torch.bool)
        record[OCCURRENCE_KEY] = occurrence
        record["_cadence_delta_t"] = float(stride / self.manifest["fps"])
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--heldout-modulo", type=int, default=5)
    args = parser.parse_args()
    try:
        rec = (prepare(args.output, source_dir=args.source_dir, heldout_modulo=args.heldout_modulo)
               if args.command == "prepare" else check(args.output))
    except (OSError, ValueError, KeyError) as err:
        parser.exit(1, f"Task7 variable cadence preparation failed: {err}\n")
    print(json.dumps({
        "protocol": rec["protocol"], "episode": rec["episode_index"],
        "train_steps": rec["steps"],
        "train_strides": rec["train_strides"],
        "pilot_holdout": len(rec["heldout_starts"]),
        "unseen_stride3": len(rec["stride3_heldout_starts"]),
        "directory": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
