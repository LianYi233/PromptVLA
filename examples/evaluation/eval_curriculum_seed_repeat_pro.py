"""Four-GPU LIBERO-PRO evaluation of D1/E1/F1 and D2/E2/F2.

Usage: bash run_eval_libero_curriculum_seed_repeat_pro.sh [run|status|summary]
Uses the same benchmark runner and result validation as the original A-H evaluation.
"""

import argparse
import json
import os
from pathlib import Path

from examples.evaluation import eval_curriculum_ablation_pro as baseline

CELLS = ("D1", "D2", "E1", "E2", "F1", "F2")


def settings(env):
    chosen = env.get("CELLS", " ".join(CELLS)).replace(",", " ").split()
    if not chosen or len(set(chosen)) != len(chosen) or any(c not in CELLS for c in chosen):
        raise ValueError("CELLS must contain unique labels from D1 D2 E1 E2 F1 F2")
    seeds = []
    for key, default in (("SEED_SET_1", 2), ("SEED_SET_2", 3)):
        value = env.get(key, str(default))
        if not value.isdigit() or not 2 <= int(value) < 2**32:
            raise ValueError(f"{key} must be an integer in [2, 2**32)")
        seeds.append(int(value))
    if seeds[0] == seeds[1]:
        raise ValueError("SEED_SET_1 and SEED_SET_2 must differ")
    if env.get("CHECKPOINT_STEP", "6000") != "6000":
        raise ValueError("These seed-repeat runs save checkpoint 6000 only")

    ckpt = Path(env.get("CKPT_ROOT", "/data1/wyn/chkpt/2601-lerobot/PromptVLA")).resolve()
    suffix = f"curriculum-def-seeds-{seeds[0]}-{seeds[1]}"
    template_env = dict(env)
    template_env.update(
        CELLS="D E F",
        REPEAT_ID="2",
        CHECKPOINT_STEP="6000",
        BASE_OUTPUT=env.get("BASE_OUTPUT", str(ckpt / "eval" / suffix)),
        EVAL_LOG_ROOT=env.get("EVAL_LOG_ROOT", f"/data1/wyn/chkpt/2601-lerobot/logs/eval/{suffix}"),
    )
    templates, batch = baseline.settings(template_env)
    templates = {job["cell"]: job for job in templates}
    gpus = env.get("GPU_IDS", "0,1,2,3").split(",")  # validated by baseline.settings
    directions = {"D": "H2E", "E": "E2H", "F": "RND"}
    jobs = []
    for index, cell in enumerate(chosen):
        template = templates[cell[0]]
        seed = seeds[int(cell[1]) - 1]
        name = (
            f"pi05-curriculum-{cell}-{directions[cell[0]]}-e50-h50-S{seed}"
            f"-steps6000-aexpert-libero-Bocc-vlm2-P{seed}-D{seed}-E{seed}-T{seed}"
        )
        gpu = gpus[index % len(gpus)]
        jobs.append(
            template
            | {
                "cell": cell,
                "name": name,
                "gpu": gpu,
                "training_seed": seed,
                "env": template["env"] | {"RUN_NAME": name, "GPU_ID": gpu},
                "checkpoint": ckpt / name / "checkpoints/006000/pretrained_model",
                "result": template["result"].parent.parent
                / f"libero060-all-{name}-006000-pro-libero-all4-resume/eval_info.json",
            }
        )
    return jobs, batch


def check_training_identity(jobs):
    """Reject misnamed/wrong-seed checkpoints before any GPU evaluation starts."""
    for job in jobs:
        path = job["checkpoint"] / "train_config.json"
        config = json.loads(path.read_text())
        factors = config.get("pdet") or {}
        seed = job["training_seed"]
        if (
            config.get("seed") != seed
            or config.get("steps") != 6000
            or config.get("batch_size") != 32
            or factors.get("enabled") is not True
            or factors.get("flow_binding") != "occurrence"
            or any(
                factors.get(key) != seed for key in ("prompt_seed", "data_seed", "noise_seed", "time_seed")
            )
        ):
            raise ValueError(f"{job['cell']}: checkpoint training seed/budget/binding mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "summary"), nargs="?", default="run")
    args = parser.parse_args()
    jobs, batch = settings(dict(os.environ))
    if args.command == "status":
        baseline.status(jobs, batch)
    elif args.command == "summary":
        raise SystemExit(0 if baseline.summary(jobs, batch) else 1)
    else:
        check_training_identity(jobs)
        raise SystemExit(0 if baseline.run(jobs, batch) else 1)


if __name__ == "__main__":
    main()
