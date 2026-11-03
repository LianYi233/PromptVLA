"""Calibrate task7 tabletop shift using Uniform ONLY and a disjoint pilot seed.

Pilot evaluates the saved 1200-step Uniform checkpoint at fixed physical
displacement magnitudes, then records the magnitude nearest 50% success.
It never tunes the displacement from Event/Time or final-evaluation results.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts import pi05_gripper_additive
from lerobot.scripts import pi05_gripper_event_experiment as experiment
from lerobot.scripts.pi05_task7_object_shift import DEFAULT_SHIFT_SEED, MAX_SHIFT_M

PROTOCOL = "task7-shift-uniform-pilot-v1"
DEFAULT_CALIBRATION = Path(
    "/data1/wyn/prompt-work/gripper-event-task7-1200/shift-calibration.json"
)
DEFAULT_PREPARATION = Path(
    "/data1/wyn/prompt-work/gripper-event-task7-1200/additive"
)


def _settings(args, shift):
    return SimpleNamespace(
        eval_root=Path(os.environ.get(
            "GRIPPER_TASK7_SHIFT_EVAL_ROOT",
            "/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/gripper-task7-object-shift",
        )),
        output_root=Path(os.environ.get(
            "OUTPUT_ROOT",
            os.environ.get("CKPT_ROOT", experiment.DEFAULT_OUTPUT),
        )),
        repeat=args.repeat,
        eval_seed=args.pilot_seed,
        episodes=args.episodes,
        only_selected_task=True,
        noise_std=0.0,
        fixed_eval=False,
        shift_m=shift,
        shift_seed=args.shift_seed,
    )


def _amplitudes(arg):
    try:
        values = tuple(float(x.strip()) for x in arg.split(","))
    except ValueError as exc:
        raise ValueError("Use comma-separated shift magnitudes, e.g. 0.02,0.04,0.06") from exc
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("Need at least two distinct calibration strengths")
    if any(not 0 < x <= MAX_SHIFT_M for x in values):
        raise ValueError(f"Pilot magnitudes must be in (0,{MAX_SHIFT_M}] meters")
    return tuple(sorted(values))


def _read_selected(path):
    data = json.loads(Path(path).read_text())
    if (
        data.get("protocol") != PROTOCOL
        or data.get("target_success_rate") != 50
        or data.get("task_id") != 7
        or data.get("episodes") != 10
        or not isinstance(data.get("results"), list)
        or not data["results"]
    ):
        raise ValueError("Unrecognized/incomplete task7 calibration record")
    selected = data.get("selected_shift_m")
    if type(selected) not in (float, int) or not 0 < selected <= MAX_SHIFT_M:
        raise ValueError("Selected displacement absent or invalid")
    if selected not in [row["shift_m"] for row in data["results"]]:
        raise ValueError("Selected shift was not evaluated")
    return data


def calibrate(args):
    magnitudes = _amplitudes(args.strengths)
    if args.pilot_seed == args.final_seed:
        raise ValueError("Pilot and final evaluation seeds MUST differ")
    if args.episodes != 10:
        raise ValueError("Focused pilot uses exactly 10 rollouts per strength")
    if args.pilot_seed < 0 or args.final_seed < 0 or args.shift_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    checkpoint_record = pi05_gripper_additive.check(args.directory)
    if (
        checkpoint_record["selection"]["task_id"] != 7
        or checkpoint_record["settings"]["steps"] != 1200
        or checkpoint_record["settings"]["base_steps"] != 1000
    ):
        raise ValueError("Calibrate only a 1000+200-step task7 experiment")
    job = experiment.build_jobs(
        checkpoint_record, args.directory, _settings(args, magnitudes[0]).output_root,
        "uniform", args.repeat
    )[0]
    if not (job["checkpoint"] / "model.safetensors").is_file():
        raise FileNotFoundError(f"Uniform checkpoint not finished: {job['checkpoint']}")
    if args.file.exists():
        existing = _read_selected(args.file)
        if (
            existing["pilot_seed"] != args.pilot_seed
            or existing["final_seed"] != args.final_seed
            or existing["shift_seed"] != args.shift_seed
            or existing["strengths_m"] != list(magnitudes)
            or existing["checkpoint"] != str(job["checkpoint"])
        ):
            raise ValueError("Different task7 shift calibration already exists; choose a new file")
        print(json.dumps(existing, indent=2))
        return existing

    outcomes = []
    root = Path(__file__).resolve().parents[3]
    for strength in magnitudes:
        run_args = [
            "bash", str(root / "run_eval_libero_gripper_task7_shift.sh"), "uniform",
            "--directory", str(args.directory),
            "--repeat", str(args.repeat),
            "--eval-seed", str(args.pilot_seed),
            "--episodes", str(args.episodes),
            "--shift-seed", str(args.shift_seed),
            "--shift-m", str(strength),
        ]
        env = {
            **os.environ,
            "GPU_ID": str(args.gpu),
            "DRY_RUN": "false",
            "GRIPPER_TASK7_SHIFT_EVAL_ROOT": str(_settings(args, strength).eval_root),
        }
        print(f"[PILOT] uniform, {strength:.3f}m, seed={args.pilot_seed}", flush=True)
        subprocess.run(run_args, env=env, cwd=root, check=True)
        setting = _settings(args, strength)
        path = experiment.result_path(job, checkpoint_record, setting)
        results = experiment.read_eval_result(path, 7, args.episodes, only_selected=True)
        outcomes.append({
            "shift_m": strength,
            "successes": results["overall_successes"],
            "episodes": args.episodes,
            "rate": results["selected_task_rate"],
            "eval_info": str(path),
        })

    best = min(outcomes, key=lambda row: (abs(row["rate"] - 50), row["shift_m"]))
    record = {
        "protocol": PROTOCOL,
        "task_id": 7,
        "target_success_rate": 50,
        "pilot_seed": args.pilot_seed,
        "final_seed": args.final_seed,
        "shift_seed": args.shift_seed,
        "episodes": args.episodes,
        "repeat": args.repeat,
        "checkpoint": str(job["checkpoint"]),
        "strengths_m": list(magnitudes),
        "results": outcomes,
        "selected_shift_m": best["shift_m"],
        "selected_pilot_success_rate": best["rate"],
        "note": "Chosen on Uniform pilot ONLY. Final seed is held out; 50% is not guaranteed.",
    }
    args.file.parent.mkdir(parents=True, exist_ok=True)
    with args.file.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, indent=2), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "selected"))
    parser.add_argument("--file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--directory", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pilot-seed", type=int, default=2000)
    parser.add_argument("--final-seed", type=int, default=1000)
    parser.add_argument("--shift-seed", type=int, default=DEFAULT_SHIFT_SEED)
    parser.add_argument("--strengths", default="0.02,0.04,0.06")
    args = parser.parse_args()
    try:
        if args.command == "selected":
            print(_read_selected(args.file)["selected_shift_m"])
        else:
            calibrate(args)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Task7 shift calibration failed: {exc}\n")


if __name__ == "__main__":
    main()
