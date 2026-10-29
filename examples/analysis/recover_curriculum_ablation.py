"""Verify completed cells and archive explicitly selected interrupted outputs."""

import argparse
import fcntl
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.scripts import pi05_curriculum_ablation as suite  # noqa: E402
from lerobot.utils.pdet import runtime_info  # noqa: E402


def recover(directory, cells):
    directory = Path(directory).resolve()
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = json.loads((directory / "jobs.json").read_text())
        statuses = json.loads((directory / "status.json").read_text())
        if not cells or len(cells) != len(set(cells)) or set(cells) - {j["cell"] for j in jobs}:
            raise ValueError("Select distinct known cells to archive")
        selected = [job for job in jobs if job["cell"] in cells]
        # Validate everything before moving any output. No signals are sent.
        for job in jobs:
            state = statuses.get(job["cell"], {})
            pid = state.get("pid")
            if pid:
                try:
                    os.killpg(int(pid), 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"Recorded process group still exists for {job['cell']}: {pid}")
            if job in selected:
                if state.get("state") not in ("failed", "interrupted"):
                    raise ValueError(f"Cell {job['cell']} is not recorded as failed/interrupted")
                run = Path(job["run"])
                if run.is_symlink() or run.parent.resolve() != Path(job["env"]["OUTPUT_ROOT"]).resolve():
                    raise ValueError("Refusing to archive an unexpected run path")
                checkpoint = run / "checkpoints" / f"{int(job['env']['FLOW_STEPS']):06d}" / "pretrained_model"
                if any(
                    (checkpoint / name).exists()
                    for name in ("config.json", "train_config.json", "model.safetensors")
                ):
                    raise ValueError(
                        f"Final model files exist for {job['cell']}; inspect instead of archiving"
                    )
        runtime = runtime_info()
        for job in jobs:
            if job not in selected and Path(job["run"]).exists():
                suite.checkpoint_complete(job)
                suite.check_current_runtime(job, runtime)
                print(f"KEEP verified completed cell {job['cell']}: {job['run']}", flush=True)
        reports = []
        for job in selected:
            run = Path(job["run"])
            if not run.exists():
                print(f"Already absent; ready to restart {job['cell']}: {run}", flush=True)
                continue
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = run.parent / ".interrupted" / f"{run.name}-{stamp}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            run.rename(destination)
            report = {
                "cell": job["cell"],
                "from": str(run),
                "archive": str(destination),
                "previous_status": statuses[job["cell"]],
                "job": job,
            }
            (destination / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
            training_log = Path(job["env"]["LOG_ROOT"]) / f"{job['name']}.log"
            if training_log.is_file():
                shutil.copy2(training_log, destination / "previous-training.log")
            reports.append(report)
            print(f"ARCHIVED {job['cell']}: {destination}", flush=True)
        print("Recovery preparation complete. Re-run the original suite with the SAME REPEAT_ID.", flush=True)
        return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/data1/wyn/chkpt/2601-lerobot/logs/curriculum-ablation-r2-steps6000"),
    )
    parser.add_argument("--archive-cell", action="append", required=True)
    args = parser.parse_args()
    recover(args.directory, args.archive_cell)
