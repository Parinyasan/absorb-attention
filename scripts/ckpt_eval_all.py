import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from sweep import run_jobs
from zeroshot import TASKS, harness_version

KINDS = {
    "zeroshot": ("zeroshot.py", "zeroshot.json"),
    "long": ("long_context.py", "long_context.json"),
    "internals": ("internals_eval.py", "internals_eval.json"),
}
INTERNALS_PARTS = "params,stats,window,quant,longdoc,rope"
SMALL_SUITES = "lm50,lm50_gate"
DEFAULT_SUITES = ("lm50,lm_main,lm_seeds,lm_confirm,lm_controls,lm_long_vanilla,lm_absorb,lm50_gate,lm_factorial,"
                  "lm_long_gated")


def load_json(path):
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    return r if isinstance(r, dict) else None


def required_lm_eval(explicit=""):
    return explicit or os.environ.get("LM_EVAL_VERSION") or harness_version()


def finished(kind, run_dir, tasks, what=(), lm_eval=None):
    r = load_json(os.path.join(run_dir, KINDS[kind][1]))
    if r is None:
        return False
    if kind == "zeroshot":
        return (r.get("limit") is None and set(tasks) <= set(r.get("tasks", {}))
                and (lm_eval is None or r.get("lm_eval") == lm_eval))
    if kind == "internals":
        return set(what) <= set(r)
    if kind == "long":
        from long_context import evict_current
        return bool(r.get("done")) and evict_current(r)
    return bool(r.get("done"))


def variant_of(run_dir):
    base = os.path.basename(run_dir)
    return base[len("lm_"):base.rindex("_s")] if base.startswith("lm_") and "_s" in base else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--suites", default=DEFAULT_SUITES)
    ap.add_argument("--what", default=INTERNALS_PARTS, help="internals: parts of internals_eval.py")
    ap.add_argument("--only", default="", help="comma list of variants to evaluate (default: all)")
    ap.add_argument("--skip_suites", default=SMALL_SUITES, help="internals: suites left out (50M models)")
    ap.add_argument("--tasks", default=TASKS, help="zeroshot tasks")
    ap.add_argument("--lm_eval_version", default=os.environ.get("LM_EVAL_VERSION", ""),
                    help="zeroshot: required harness version (default $LM_EVAL_VERSION, else the installed one); "
                         "refused if the installed version differs")
    ap.add_argument("--args", default="", help="extra arguments for the per-checkpoint script")
    ap.add_argument("--gpus", default="0", help="comma list of GPU indices, e.g. 0,1,2,3")
    ap.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    script = os.path.join("scripts", KINDS[a.kind][0])
    tasks = a.tasks.split(",")
    jobs = []
    only = set(a.only.split(",")) if a.only else None
    what = [w for w in a.what.split(",") if w]
    lm_eval = None
    if a.kind == "zeroshot":
        have = harness_version()
        if a.lm_eval_version and have != a.lm_eval_version:
            sys.exit(f"lm-eval {have or '(not installed)'} is installed but {a.lm_eval_version} is required "
                     f"(--lm_eval_version / LM_EVAL_VERSION): refusing to run")
        lm_eval = required_lm_eval(a.lm_eval_version)
        print("zero-shot results count as finished " + (f"only if scored with lm-eval {lm_eval}" if lm_eval else
              "whatever their harness version (lm-eval is not installed, none is required)"), flush=True)
    need_longdoc = False
    slow = set()
    for suite in a.suites.split(","):
        if a.kind == "internals" and suite in a.skip_suites.split(","):
            continue
        for ck in sorted(glob.glob(os.path.join("runs", suite, "*", "ckpt.pt"))):
            run_dir = os.path.dirname(ck)
            if only is not None and variant_of(run_dir) not in only:
                continue
            if finished(a.kind, run_dir, tasks, what, lm_eval=lm_eval):
                continue
            cmd = [sys.executable, script, "--ckpt", ck]
            if a.kind == "zeroshot":
                cmd += ["--tasks", a.tasks] + (["--require_lm_eval", lm_eval] if lm_eval else [])
            if a.kind == "internals":
                cmd += ["--what", ",".join(what)]
                need_longdoc |= "longdoc" in what and not finished(a.kind, run_dir, tasks, ["longdoc"])
            tag = a.kind + ("-" + "+".join(what) if a.kind == "internals" else "")
            name = f"{tag}_{suite}_{os.path.basename(run_dir)}"
            if a.kind == "internals" and "evict" in what and len(what) > 1 and not finished(a.kind, run_dir, tasks, ["evict"]):
                slow.add(name)
            jobs.append((name, cmd + a.args.split(), os.path.join("runs", suite)))
    jobs.sort(key=lambda j: j[0] not in slow)
    print(f"{len(jobs)} checkpoints to evaluate ({a.kind})" + (f", {len(slow)} of them with 'evict' still to do "
                                                               f"(started first)" if slow else ""), flush=True)
    if a.dry:
        for n, c, _ in jobs:
            print(" ".join(c))
        return
    if not jobs:
        return
    pre = None
    if a.kind == "zeroshot":
        pre = [sys.executable, script, "--prefetch", "--tasks", a.tasks]
    elif need_longdoc:
        pre = [sys.executable, script, "--prefetch"] + a.args.split()
    if pre is not None:
        rc = subprocess.run(pre, check=False).returncode
        if rc != 0:
            print(f"prefetch exited {rc}; continuing", flush=True)
    run_jobs(jobs, a.gpus, a.per_gpu, retries=1)


if __name__ == "__main__":
    main()
