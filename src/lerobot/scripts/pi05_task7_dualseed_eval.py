"""Two-stage task7 difficulty-calibrated, independent scene/policy-seed evaluation.

CALIBRATE: evaluate ONLY the existing trained 1000-step RoPE checkpoint
on a predeclared scene-seed pool and calibration-policy-seed pool. If its
POOLED success is 40..50%, write an immutable configuration lock.

COMPARE: on the SAME locked scene pool, evaluate all selected models with
predeclared, disjoint held-out policy RNG seeds. Optionally add fresh,
unselected scene seeds as a SEPARATE generalization report. Never retune
after viewing a comparison; never guarantee validation baseline=40..50%.

Do NOT pool calibration episodes with held-out comparison episodes.
No checkpoint is retrained or modified by this script.
"""

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts import pi05_task7_timepath_experiment as original
from lerobot.scripts import pi05_task7_input_spectrum as spectral
from lerobot.scripts import pi05_gripper_event_experiment as previous
from lerobot.scripts import pi05_task7_dualseed_shift as dual

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "pi05-task7-dualseed-calibration-lock-v1"
CASES = ("rope", "time_qk", "time_v", "time_input",
         "lowfreq", "rms_gate", "lowfreq_rms")
DEFAULT_CASES = ("rope", "time_input", "lowfreq", "rms_gate", "lowfreq_rms")
DEFAULT_ROOT = original.DEFAULT_OUTPUT / "eval" / "task7-dualseed-locked"
DEFAULT_CAL_SCENES = "20261103,20261104"
DEFAULT_CAL_POLICY = "101,102"
DEFAULT_TEST_POLICY = "1001,1002,1003"


def seeds(value, label):
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated nonnegative integers") from exc
    if (
        not result or len(result) != len(set(result))
        or any(value < 0 or value >= 2**31 for value in result)
    ):
        raise ValueError(f"{label} needs distinct nonnegative integers < 2^31")
    return result


def case_names(value):
    if value == "all":
        return CASES
    result = tuple(value.split(","))
    if not result or any(v not in CASES for v in result) or len(set(result)) != len(result):
        raise ValueError("Cases must be unique names from: " + ",".join(CASES))
    if "rope" not in result:
        raise ValueError("RoPE baseline must be present in every comparison")
    return result


def job_map(record, args):
    old = original.build_jobs(record, args.directory, args.output_root, "all", args.repeat)
    new = spectral.training_jobs(record, args, "all")
    return {x["case"]: x for x in [*old, *new]}


def is_original(case):
    return case in original.CASES


def eval_arguments(args, case, scene_seed, policy_seed):
    return SimpleNamespace(
        eval_root=(args.control_eval_root if is_original(case) else args.optimized_eval_root),
        eval_seed=args.init_seed, shift_m=args.shift_m,
        shift_seed=20261103,  # ignored under dual protocol
        scene_seed=int(scene_seed), policy_seed=int(policy_seed),
        repeat=args.repeat, episodes=10,
    )


def fingerprint(checkpoint):
    checkpoint=Path(checkpoint)
    files=[
        [str(p.relative_to(checkpoint)), p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(checkpoint.rglob("*")) if p.is_file()
    ]
    if not files or not (checkpoint/"model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model files: {checkpoint}")
    return files


def run_location(record, args, job, scene_seed, policy_seed):
    opts=eval_arguments(args,job["case"],scene_seed,policy_seed)
    return original.result_path(job,record,opts)


def _raw_per_episode(path):
    info=json.loads(Path(path).read_text())
    entries=info.get("per_task",info.get("Aggregated Metrics for per_task",[]))
    if not isinstance(entries,list) or len(entries)!=1:
        raise ValueError(f"Expected one task7 record in {path}")
    task=entries[0]
    if task.get("task_group")!="libero_10" or task.get("task_id")!=7:
        raise ValueError(f"Not standard libero_10 task7: {path}")
    flags=task.get("metrics",task).get("successes")
    if not isinstance(flags,list) or len(flags)!=10 or any(type(v) not in (bool,int)
                                                    or v not in (0,1) for v in flags):
        raise ValueError(f"Incomplete/invalid task7 outcome flags: {path}")
    return [bool(v) for v in flags]


def verified_result(record, args, job, scene_seed, policy_seed):
    opts=eval_arguments(args,job["case"],scene_seed,policy_seed)
    path=run_location(record,args,job,scene_seed,policy_seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    config_path=original.evaluation_config_path(opts,record)
    config=original.shift_config(opts)
    if json.loads(config_path.read_text())!=config:
        raise ValueError(f"Dual-seed config changed: {config_path}")
    manifest=json.loads((path.parent/"resume_manifest.json").read_text())
    intervention=manifest.get("task7_dualseed_shift",{})
    if (
        intervention.get("config")!=config
        or intervention.get("config_sha256")!=previous.file_hash(config_path)
        or intervention.get("helper_sha256")!=previous.file_hash(Path(dual.__file__))
        or manifest.get("fixed_observation_noise")
        or manifest.get("task7_object_shift")
        or manifest.get("model_files")!=fingerprint(job["checkpoint"])
    ):
        raise ValueError(
            "Cached evaluation differs from dual-seed scene/policy/model protocol: "+str(path)
        )
    flags=_raw_per_episode(path)
    return {
        "scene_seed":int(scene_seed),
        "policy_seed":int(policy_seed),
        "case":job["case"],
        "successes":int(sum(flags)),
        "episodes":10,
        "per_episode":flags,
        "eval_info":str(path),
        "checkpoint":str(job["checkpoint"]),
    }


def evaluation_command(record,args,job,scene_seed,policy_seed,gpu):
    module=(
        "lerobot.scripts.pi05_task7_timepath_experiment"
        if is_original(job["case"])
        else "lerobot.scripts.pi05_task7_input_spectrum"
    )
    subcommand="eval" if is_original(job["case"]) else "eval_one"
    opts=eval_arguments(args,job["case"],scene_seed,policy_seed)
    return [
        "python","-u","-m",module,subcommand,job["case"],
        "--directory",str(args.directory),
        "--output-root",str(args.output_root),
        "--repeat",str(args.repeat),
        "--eval-root",str(opts.eval_root),
        "--tokenizer",str(args.tokenizer),
        "--eval-seed",str(args.init_seed),
        "--episodes","10",
        "--shift-m",str(args.shift_m),
        "--scene-seed",str(scene_seed),
        "--policy-seed",str(policy_seed),
        "--gpu",str(gpu),
    ]


def evaluate_one(record,args,job,scene_seed,policy_seed,gpu):
    path=run_location(record,args,job,scene_seed,policy_seed)
    if path.exists():
        return verified_result(record,args,job,scene_seed,policy_seed)
    command=evaluation_command(record,args,job,scene_seed,policy_seed,gpu)
    args.log_dir.mkdir(parents=True,exist_ok=True)
    log=args.log_dir/f"scene{scene_seed}-policy{policy_seed}-{job['case']}-gpu{gpu}.log"
    env=os.environ.copy()
    for key in ("LIBERO_TASK7_OBJECT_SHIFT_CONFIG",
                "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG",
                "LIBERO_ROBUSTNESS_MANIFEST","LIBERO_BENCHMARK",
                "LIBERO_FIXED_NOISE_CONFIG","PI05_CADENCE_PLAN",
                "CUDA_VISIBLE_DEVICES"):
        env.pop(key,None)
    env["GPU_ID"]=str(gpu)
    env["DRY_RUN"]="false"
    env["PYTHONPATH"]=str(ROOT/"src")+":"+env.get("PYTHONPATH","")
    print(
        f"GPU {gpu}: {job['case']} scene={scene_seed} policy={policy_seed}; log={log}",
        flush=True,
    )
    with log.open("w") as stream:
        completed=subprocess.run(
            command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"Failed {job['case']} scene={scene_seed} policy={policy_seed}: {log}")
    return verified_result(record,args,job,scene_seed,policy_seed)


def precreate_configs(record,args,jobs,scene_seed,policy_seed):
    # Do not allow two parallel workers to race on writing the same JSON.
    for case in set("original" if is_original(j["case"]) else "new" for j in jobs):
        sample=next(job for job in jobs if is_original(job["case"])==(case=="original"))
        opts=eval_arguments(args,sample["case"],scene_seed,policy_seed)
        previous.ensure_noise_config(
            original.evaluation_config_path(opts,record),original.shift_config(opts)
        )


def evaluate_pair(record,args,jobs,scene_seed,policy_seed):
    precreate_configs(record,args,jobs,scene_seed,policy_seed)
    gpu_ids=[int(x) for x in args.gpus.split(",")]
    rows={}
    for start in range(0,len(jobs),2):
        batch=jobs[start:start+2]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures=[
                pool.submit(evaluate_one,record,args,job,scene_seed,policy_seed,gpu_ids[index])
                for index,job in enumerate(batch)
            ]
            for job,future in zip(batch,futures,strict=True):
                row=future.result()
                rows[job["case"]]=row
    return rows


def configuration(args,record,baseline):
    return {
        "protocol":PROTOCOL,
        "selected_task":"libero_10/task7",
        "training_episode":record["selection"]["episode_index"],
        "preparation_sha256":previous.file_hash(args.directory/"manifest.json"),
        "baseline_checkpoint":str(baseline["checkpoint"]),
        "baseline_checkpoint_files":fingerprint(baseline["checkpoint"]),
        "scene_seeds":list(args.scene_seeds),
        "calibration_policy_seeds":list(args.cal_policy_seeds),
        "heldout_policy_seeds":list(args.test_policy_seeds),
        "heldout_scene_seeds":list(args.external_scene_seeds),
        "init_seed":args.init_seed,
        "shift_m":float(args.shift_m),
        "target_success_band_pct":[float(args.target_min),float(args.target_max)],
        "selection":"All scene seeds predeclared; no post hoc seed subset selection.",
    }


def settings_hash(settings):
    return hashlib.sha256(
        json.dumps(settings,sort_keys=True).encode()
    ).hexdigest()[:12]


def summarize_calibration(rows, min_pct, max_pct):
    if not rows:
        raise ValueError("No completed RoPE calibration")
    hits=sum(r["successes"] for r in rows)
    count=sum(r["episodes"] for r in rows)
    rate=100*hits/count
    return {
        "successes":hits,"episodes":count,
        "rate_percent":rate,"in_target_band":min_pct<=rate<=max_pct,
    }


def calibrate(record,args,models):
    settings=configuration(args,record,models["rope"])
    if args.lock_file.exists():
        stored=verify_lock(args,record,models)
        print(f"Already locked: {args.lock_file}; rate={stored['calibration']['rate_percent']:.1f}%")
        return stored
    all_jobs=[]
    for scene in args.scene_seeds:
        for policy in args.cal_policy_seeds:
            all_jobs.append((scene,policy))
    results=[]
    print(f"Calibrating ONLY RoPE on {len(all_jobs)*10} episodes; shift={args.shift_m:.3f}m")
    gpu_ids=[int(x) for x in args.gpus.split(",")]
    # Each scene/policy pair has its own immutable eval directory. Schedule
    # two distinct calibration cells concurrently across both GPUs.
    for start in range(0,len(all_jobs),2):
        cells=all_jobs[start:start+2]
        for scene,policy in cells:
            precreate_configs(record,args,[models["rope"]],scene,policy)
        with ThreadPoolExecutor(max_workers=len(cells)) as pool:
            futures=[
                pool.submit(
                    evaluate_one,record,args,models["rope"],scene,policy,gpu_ids[i]
                )
                for i,(scene,policy) in enumerate(cells)
            ]
            for (scene,policy),future in zip(cells,futures,strict=True):
                result=future.result()
                results.append(result)
                print(f"  scene={scene} policy={policy}  RoPE {result['successes']}/10")
    status=summarize_calibration(results,args.target_min,args.target_max)
    args.report_root.mkdir(parents=True,exist_ok=True)
    saved={
        "settings":settings,"calibration":status,"calibration_rows":results,
        "selection_bias_warning":(
            "This stress suite is calibrated ON RoPE success; it is not an "
            "unselected population benchmark. Validation policy seeds remain unused."
        ),
    }
    path=args.report_root/f"calibration-{settings_hash(settings)}.json"
    path.write_text(json.dumps(saved,indent=2,sort_keys=True)+"\n")
    print(f"RoPE CALIBRATION = {status['successes']}/{status['episodes']} "
          f"({status['rate_percent']:.1f}%), target=[{args.target_min},{args.target_max}]")
    if not status["in_target_band"]:
        print(
            "OUTSIDE TARGET: no lock written. Do not cherry-pick individual scene seeds. "
            "Adjust task difficulty only on calibration data, record changes, and "
            "reconfirm on untouched held-out policy seeds.",
            flush=True,
        )
        raise ValueError(
            f"RoPE calibration {status['rate_percent']:.1f}% is outside "
            f"[{args.target_min},{args.target_max}]%; calibration report saved at {path}; "
            "configuration NOT locked."
        )
    args.lock_file.parent.mkdir(parents=True,exist_ok=True)
    with args.lock_file.open("x") as f:
        json.dump(saved,f,indent=2,sort_keys=True)
        f.write("\n")
    print(f"LOCKED calibrated scenario pool: {args.lock_file}",flush=True)
    return saved


def verify_lock(args,record,models):
    if not args.lock_file.is_file():
        raise FileNotFoundError(
            f"No successful RoPE calibration lock: {args.lock_file}. Run calibrate first."
        )
    lock=json.loads(args.lock_file.read_text())
    expected=configuration(args,record,models["rope"])
    if lock.get("settings")!=expected:
        raise ValueError(
            "Locked scene/policy seeds, task data, checkpoint or difficulty changed. "
            "Refusing to tune test conditions after calibration."
        )
    if not lock.get("calibration",{}).get("in_target_band"):
        raise ValueError("RoPE calibration was not in 40–50% target band")
    observed=[]
    for scene in args.scene_seeds:
        for policy in args.cal_policy_seeds:
            observed.append(verified_result(record,args,models["rope"],scene,policy))
    if observed!=lock["calibration_rows"]:
        raise ValueError("Locked RoPE calibration results changed")
    return lock


def paired_statistics(rope,other):
    a=rope["per_episode"]
    b=other["per_episode"]
    wins=sum((not x) and y for x,y in zip(a,b,strict=True))
    losses=sum(x and (not y) for x,y in zip(a,b,strict=True))
    return {"model_only_wins":wins,"rope_only_wins":losses,"net_wins":wins-losses}


def render_table(rows,cases,stage):
    print("\n"+("="*110))
    print(f"{stage} | fixed init states 0..9; locked shift | 10 episodes per cell")
    labels=["Scene","Policy",*cases]
    widths=[max(len(x),12) for x in labels]
    def fmt(xs):
        return "  ".join(str(value).ljust(widths[i]) for i,value in enumerate(xs))
    print(fmt(labels))
    print("  ".join("-"*w for w in widths))
    aggregate={case:[0,0] for case in cases}
    pairs={case:[0,0] for case in cases if case!="rope"}
    for row in rows:
        this=[str(row["scene_seed"]),str(row["policy_seed"])]
        for case in cases:
            result=row["results"][case]
            hits=result["successes"]
            aggregate[case][0]+=hits
            aggregate[case][1]+=10
            this.append(f"{hits}/10")
            if case!="rope":
                matched=paired_statistics(row["results"]["rope"],result)
                pairs[case][0]+=matched["model_only_wins"]
                pairs[case][1]+=matched["rope_only_wins"]
        print(fmt(this))
    print(fmt(["POOLED","",*[f"{hits}/{n}={100*hits/n:.1f}%"
                           for hits,n in (aggregate[c] for c in cases)]]))
    for case in cases:
        if case=="rope":
            continue
        wins,losses=pairs[case]
        print(f"  {case} vs RoPE | model-only wins={wins}, "
              f"RoPE-only wins={losses}, net={wins-losses:+d} episodes")
    return {"aggregate":aggregate,"paired_counts":pairs}


def report_path(args,lock,stage,cases):
    settings={
        "lock_sha256":previous.file_hash(args.lock_file),
        "stage":stage,
        "cases":list(cases),
    }
    return args.report_root/f"comparison-{stage}-{settings_hash(settings)}.json"


def compare(record,args,models,*,execute):
    lock=verify_lock(args,record,models)
    cases=case_names(args.cases)
    for case in cases:
        fingerprint(models[case]["checkpoint"])
    evaluations=[
        ("calibrated_stress",args.scene_seeds)
    ]
    if args.external_scene_seeds:
        evaluations.append(("unselected_scenes",args.external_scene_seeds))
    for stage,scene_seeds in evaluations:
        cells=[]
        for scene in scene_seeds:
            for policy in args.test_policy_seeds:
                if execute:
                    results=evaluate_pair(
                        record,args,[models[case] for case in cases],scene,policy
                    )
                else:
                    results={
                        case:verified_result(record,args,models[case],scene,policy)
                        for case in cases
                    }
                cells.append({
                    "scene_seed":scene,"policy_seed":policy,"results":results,
                })
        summary=render_table(cells,cases,stage)
        report={
            "protocol":PROTOCOL,"stage":stage,"lock":lock,
            "cases":list(cases),"evaluations":cells,
            "summary":summary,
            "note":(
                "Baseline success band was selected on calibration draws only. "
                "Validation RoPE success may be outside 40–50%; NEVER retroactively "
                "reselect scene/policy seeds after viewing intervention outcomes."
            ),
        }
        path=report_path(args,lock,stage,cases)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        with path.with_suffix(".csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["stage","scene_seed","policy_seed","case",
                             "successes","episodes","rate_pct"])
            for cell in cells:
                for case in cases:
                    n=cell["results"][case]["successes"]
                    writer.writerow([
                        stage,cell["scene_seed"],cell["policy_seed"],
                        case,n,10,n*10.0
                    ])
        print(f"Saved comparison: {path}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("calibrate","compare","report"))
    parser.add_argument("--scene-seeds",default=DEFAULT_CAL_SCENES)
    parser.add_argument("--cal-policy-seeds",default=DEFAULT_CAL_POLICY)
    parser.add_argument("--test-policy-seeds",default=DEFAULT_TEST_POLICY)
    parser.add_argument("--external-scene-seeds",default="")
    parser.add_argument("--cases",default=",".join(DEFAULT_CASES))
    parser.add_argument("--target-min",type=float,default=40.0)
    parser.add_argument("--target-max",type=float,default=50.0)
    parser.add_argument("--shift-m",type=float,default=0.12)
    parser.add_argument("--init-seed",type=int,default=3000)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--gpus",default=os.environ.get("GPU_IDS","0,1"))
    parser.add_argument("--directory",type=Path,default=original.DEFAULT_PLAN)
    parser.add_argument("--output-root",type=Path,default=Path(os.environ.get(
        "OUTPUT_ROOT",original.DEFAULT_OUTPUT,
    )))
    parser.add_argument("--control-eval-root",type=Path,default=original.DEFAULT_EVAL)
    parser.add_argument("--optimized-eval-root",type=Path,default=spectral.DEFAULT_EVAL)
    parser.add_argument("--tokenizer",type=Path,default=Path(os.environ.get(
        "TOKENIZER_PATH","/home/wyn/models/paligemma-3b-pt-224"
    )))
    parser.add_argument("--report-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--lock-file",type=Path,default=None)
    parser.add_argument("--log-dir",type=Path,default=None)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    try:
        args.scene_seeds=seeds(args.scene_seeds,"scene seeds")
        args.cal_policy_seeds=seeds(args.cal_policy_seeds,"calibration policy seeds")
        args.test_policy_seeds=seeds(args.test_policy_seeds,"heldout policy seeds")
        args.external_scene_seeds=(
            seeds(args.external_scene_seeds,"unselected scene seeds")
            if args.external_scene_seeds else ()
        )
        if set(args.cal_policy_seeds)&set(args.test_policy_seeds):
            raise ValueError("Calibration and heldout policy seeds must be disjoint")
        if set(args.scene_seeds)&set(args.external_scene_seeds):
            raise ValueError("Unselected scene seeds cannot overlap calibrated seeds")
        gpus=seeds(args.gpus,"GPU indices")
        if (
            len(gpus)!=2 or args.repeat<1 or args.init_seed<0
            or args.init_seed>=2**31-10
            or not 0<args.shift_m<=0.14
            or not 0<=args.target_min<=args.target_max<=100
        ):
            raise ValueError("Require two different GPUs, fixed init seed, valid magnitude/target")
        args.gpus=",".join(map(str,gpus))
        args.report_root=args.report_root.resolve()
        args.directory=args.directory.resolve()
        args.lock_file=(args.lock_file or args.report_root/"calibration-lock.json").resolve()
        args.log_dir=(args.log_dir or args.report_root/"logs").resolve()
        if args.dry_run:
            jobs=[]
            if args.command=="calibrate":
                for scene in args.scene_seeds:
                    for policy in args.cal_policy_seeds:
                        jobs.append(f"CAL: scene={scene}, policy={policy}, rope only")
            else:
                for scene in (*args.scene_seeds,*args.external_scene_seeds):
                    for policy in args.test_policy_seeds:
                        jobs.append(f"TEST: scene={scene}, policy={policy}, cases={args.cases}")
            print("\n".join(jobs))
            print("Dry-run: no GPU allocations, checkpoint loading or results written")
            return
        record=original.check_plan(args.directory)
        models=job_map(record,args)
        if args.command=="calibrate":
            fingerprint(models["rope"]["checkpoint"])
            calibrate(record,args,models)
        else:
            compare(record,args,models,execute=(args.command=="compare"))
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        parser.exit(1,f"Task7 dual-seed locked evaluation failed: {exc}\n")


if __name__=="__main__":
    main()
