import argparse
import itertools
import os
import subprocess
import sys
import time

from presets import ABLATIONS, COUPLING

SYN = {
    "mqar": ("mqar", 64, ["n_keys=8", "writes=2", "n_queries=8"]),
    "vt": ("vt", 128, ["n_vars=4"]),
    "niah": ("niah", 128, ["n_needles=4", "n_queries=2"]),
}
SYN_HARD = {
    "mqar_h": ("mqar", 128, ["n_keys=8", "writes=4", "n_queries=8"]),
    "vt_h": ("vt", 256, ["n_vars=8"], "--batch 128"),
    "niah_h": ("niah", 512, ["n_needles=6", "n_queries=3"], "--batch 64"),
}
LM = {"lm": ("lm", 1024, [])}

MAIN = ["vanilla", "gated", "gated_head", "sink", "selective", "fox", "stickbreak", "ddl",
        "absorb", "absorb_shared", "absorb_odds0"]
SA_FAMILY = ["sa_null", "sa_null_delta", "sa_null_delta_free", "sa_constnull_delta", "sa_delta", "absorb_odds0_h0"]
SYN_COMMON = "--model tiny --batch 256 --lr 1e-3 --eval_every 500 --eval_batches 4 --eval_scales 1,2,4"
LM100 = ("--model lm100m --steps 20000 --batch 16 --accum 4 --lr 6e-4 --warmup 0.02 "
         "--eval_every 2000 --eval_batches 20 --eval_scales 1,2,4 --save_ckpt --grad_ckpt")
LM_CONFIRM_RUNS = [
    ("selective", 1), ("sa_null", 0), ("sa_delta", 0), ("absorb_odds0", 0),
    ("selective", 2), ("sa_null", 1), ("sa_delta", 1), ("absorb_odds0", 1),
    ("sa_null", 2), ("sa_delta", 2), ("absorb_odds0", 2),
    ("gated", 1), ("gated", 2), ("fox", 1), ("fox", 2), ("sink", 1), ("sink", 2),
]

LM100C = LM100.replace(" --grad_ckpt", "") + " --compile"
CALIB_RUNS = [("sa_null", 0), ("selective", 0)]
LM_CONTROL_RUNS = (
    CALIB_RUNS
    + [(v, s) for s in (0, 1, 2) for v in ("sa_constnull", "sa_constnull0")]
    + [(v, s) for s in (3, 4) for v in ("selective", "sa_null", "sa_delta", "sa_null_delta")]
    + [("sink0", s) for s in (0, 1, 2)]
    + [(v, s) for s in (0, 1, 2) for v in ("sa_constnull_delta", "sa_constnull0_delta")]
    + [("sa_null_gain", s) for s in (0, 1, 2)]
    + [("ddl_attn", s) for s in (0, 1, 2)] + [("ddl", 0)]
    + [(v, s) for s in (0, 1, 2) for v in ("selective_bos", "sa_null_delta_bos")]
    + [("sa_g1", 0), ("sa_g1_head", 0), ("absorb_null", 0)]
    + [("sa_constnull", s) for s in (3, 4)]
    + [(v, s) for s in (1, 2) for v in ("sa_g1", "sa_g1_head")]
    + [("absorb_g1", s) for s in (0, 1, 2)]
)


def with_args(spec, args):
    return spec[:3] + ((spec[3] + " " if len(spec) > 3 else "") + args,)


RATE_VARIANTS = ["absorb_rho1", "absorb_shared_rho6", "absorb_scaled"]
LRS = ["1e-4", "4.6e-4", "1e-3", "2.2e-3", "1e-2"]
LR_TASKS = {f"{t}_lr{lr}": with_args(SYN[t], f"--lr {lr}") for t in ("mqar", "vt", "niah") for lr in LRS}
LR_VARIANTS = ["vanilla", "sink0", "gated", "selective", "fox", "stickbreak", "sa_null_delta", "sa_null", "sa_delta",
               "sa_constnull", "absorb", "absorb_shared"]
LR_REUSE = {f"{t}_lr1e-3": ("syn_tasks", t) for t in ("vt", "niah")}

SUITES = {
    "syn_mqar": dict(tasks={"mqar": SYN["mqar"], "mqar_h": SYN_HARD["mqar_h"]},
                     variants=MAIN + SA_FAMILY, seeds=list(range(10)), common=SYN_COMMON + " --steps 6000"),
    "syn_tasks": dict(tasks={"vt": SYN["vt"], "niah": SYN["niah"], "vt_h": SYN_HARD["vt_h"],
                             "niah_h": SYN_HARD["niah_h"]},
                      variants=MAIN + SA_FAMILY, seeds=list(range(5)),
                      common=SYN_COMMON + " --steps 20000 --evict_eval"),
    "syn_ablate": dict(tasks={"mqar": SYN["mqar"], "vt": SYN["vt"], "niah": SYN["niah"]},
                       variants=ABLATIONS[1:] + COUPLING + ["absorb_readgate", "absorb_iter2", "sink0", "fox_rope"],
                       seeds=list(range(5)), common=SYN_COMMON + " --steps 6000"),
    "lm50": dict(tasks=LM, variants=MAIN + ["sa_null", "sa_null_delta"],
                 seeds=[0], common="--model lm50m --steps 8000 --batch 16 --accum 4 --lr 1e-3 --warmup 0.03 "
                                   "--eval_every 1000 --eval_batches 16 --eval_scales 1,2,4 --save_ckpt --evict_eval"),
    "lm_main": dict(tasks=LM, variants=["vanilla", "gated", "sink", "selective", "fox", "sa_null_delta"],
                    seeds=[0], common=LM100),
    "lm_seeds": dict(tasks=LM, variants=["vanilla", "sa_null_delta"], seeds=[1, 2], common=LM100),
    "lm_confirm": dict(tasks=LM, runs=LM_CONFIRM_RUNS, common=LM100),
    "lm_controls": dict(tasks=LM, runs=LM_CONTROL_RUNS, common=LM100C),
    "lm_long_vanilla": dict(tasks=LM, runs=[("vanilla", 0)],
                            common=LM100C.replace("--steps 20000", "--steps 32000").replace("--eval_every 2000",
                                                                                           "--eval_every 4000")),
    "syn_rate": dict(tasks={"mqar": SYN["mqar"]}, variants=RATE_VARIANTS, seeds=list(range(10)),
                     common=SYN_COMMON + " --steps 6000"),
    "syn_rate_long": dict(tasks={"mqar_20k": SYN["mqar"]}, variants=RATE_VARIANTS + ["absorb", "absorb_shared"],
                          seeds=list(range(10)),
                          common=SYN_COMMON + " --steps 20000"),
    "syn_rate_tasks": dict(tasks={"vt": SYN["vt"], "niah": SYN["niah"]}, variants=RATE_VARIANTS, seeds=list(range(5)),
                           common=SYN_COMMON + " --steps 20000"),
    "syn_lr": dict(tasks=LR_TASKS, variants=LR_VARIANTS, seeds=[0, 1, 2], common=SYN_COMMON + " --steps 20000",
                   reuse=LR_REUSE),
}

LM_ABSORB_RUNS = [("selective_wide", 0), ("sa_g1", 3), ("selective_wide", 1), ("sa_g1", 4), ("selective_wide", 2)]
LM50_GATE_VARIANTS = ["sa_g1", "sa_g1_head", "absorb_g1"]
LM_FACTORIAL_RUNS = [("gated", 3), ("vanilla", 3), ("gated", 4), ("vanilla", 4)]
SUITES.update({
    "lm_absorb": dict(tasks=LM, runs=LM_ABSORB_RUNS, common=LM100C),
    "lm50_gate": dict(tasks=LM, variants=LM50_GATE_VARIANTS, seeds=[0], common=SUITES["lm50"]["common"]),
    "lm_factorial": dict(tasks=LM, runs=LM_FACTORIAL_RUNS, common=LM100C),
})

SUITES["lm_long_gated"] = dict(tasks=LM, runs=[("gated", 0)], common=SUITES["lm_long_vanilla"]["common"])


def suite_runs(s):
    return s["runs"] if "runs" in s else list(itertools.product(s["variants"], s["seeds"]))


def check_spec(tname, spec):
    assert isinstance(spec, tuple) and len(spec) in (3, 4), f"task spec {tname}: {spec!r} is not a 3/4-tuple"
    assert spec[0] in ("mqar", "vt", "niah", "lm") and isinstance(spec[1], int), f"task spec {tname}: {spec!r}"
    assert isinstance(spec[2], list) and all(isinstance(x, str) and "=" in x for x in spec[2]), \
        f"task spec {tname}: task kwargs {spec[2]!r} must be a list of key=value strings"
    assert len(spec) == 3 or isinstance(spec[3], str), f"task spec {tname}: extra args {spec[3]!r} must be a string"


def check_suites():
    for name, s in SUITES.items():
        assert ("runs" in s) != ("variants" in s and "seeds" in s), f"suite {name}: give `runs` or variants+seeds"
        for tname, spec in s["tasks"].items():
            check_spec(tname, spec)
        for tname, (src, src_t) in s.get("reuse", {}).items():
            assert s["tasks"][tname][:3] == SUITES[src]["tasks"][src_t][:3], f"suite {name}: reuse {tname}"
    for t in ("mqar", "vt", "niah"):
        for lr in LRS:
            assert LR_TASKS[f"{t}_lr{lr}"][-1].endswith(f"--lr {lr}"), f"syn_lr {t} {lr}: lr not applied"


check_suites()


def reused_runs(suite, out):
    s = SUITES[suite]
    found = {}
    for tname, (src, src_t) in s.get("reuse", {}).items():
        have = set(suite_runs(SUITES[src]))
        for var, seed in suite_runs(s):
            if (var, seed) in have:
                found[f"{tname}_{var}_s{seed}"] = os.path.join(os.path.dirname(os.path.normpath(out)), src,
                                                               f"{src_t}_{var}_s{seed}", "results.json")
    return found


def jobs_for(suite, out, extra, data_dir):
    s = SUITES[suite]
    reuse = {n for n, p in reused_runs(suite, out).items() if os.path.exists(p)}
    for (tname, spec), (var, seed) in itertools.product(s["tasks"].items(), suite_runs(s)):
        check_spec(tname, spec)
        task, L, tkw = spec[:3]
        task_extra = spec[3] if len(spec) > 3 else ""
        name = f"{tname}_{var}_s{seed}"
        if name in reuse:
            continue
        cmd = [sys.executable, "train.py", "--task", task, "--variant", var, "--seed", str(seed),
               "--seq_len", str(L), "--out", out, "--name", name] + s["common"].split() + task_extra.split() + extra.split()
        if task == "lm":
            cmd += ["--data_dir", data_dir]
        if tkw:
            cmd += ["--tkw"] + tkw
        yield name, cmd


def pick_gpu(busy, per_gpu):
    cands = [g for g, b in busy.items() if b < per_gpu]
    for g in sorted(cands, key=lambda g: (busy[g], int(g))):
        return g
    return None


def run_jobs(todo, gpus="0", per_gpu=1, retries=1):
    todo = list(todo)
    if not todo:
        return
    pool = [g for g in gpus.split(",") if g]
    busy = {g: 0 for g in pool}
    print(f"[gpus] using {','.join(pool)} x {per_gpu} job(s) each", flush=True)
    running = {}
    attempts = {}
    while todo or running:
        while todo:
            gpu = pick_gpu(busy, per_gpu)
            if gpu is None:
                break
            job = todo.pop(0)
            n, c, out = job
            attempts[n] = attempts.get(n, 0) + 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            os.makedirs(os.path.join(out, "_logs"), exist_ok=True)
            log = open(os.path.join(out, "_logs", f"{n}.log" if attempts[n] == 1 else f"{n}.try{attempts[n]}.log"), "w")
            running[subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT, env=env)] = (gpu, job, log)
            busy[gpu] += 1
            print(f"[start] {n} on gpu {gpu} ({len(todo)} queued)", flush=True)
        for p in [p for p in running if p.poll() is not None]:
            gpu, job, log = running.pop(p)
            log.close()
            busy[gpu] -= 1
            n = job[0]
            if p.returncode != 0 and attempts[n] <= retries:
                todo.append(job)
                print(f"[failed rc={p.returncode}] {n} -> retry {attempts[n]}/{retries} later", flush=True)
            else:
                print(f"[done rc={p.returncode}] {n}", flush=True)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    help="suite name, or a comma list (e.g. lm_main,lm_seeds) scheduled together")
    ap.add_argument("--out", default=None, help="output dir (single suite only; default runs/<suite>)")
    ap.add_argument("--gpus", default="0", help="comma list of GPU indices, e.g. 0,1,2,3")
    ap.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--retries", type=int, default=1, help="re-queue a job that exits with an error this many times")
    ap.add_argument("--extra", default="", help="extra args appended to every train.py call")
    ap.add_argument("--only", default="", help="comma list of variants to keep")
    ap.add_argument("--data_dir", default="data/fineweb_edu")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    suites = a.suite.split(",")
    unknown = [x for x in suites if x not in SUITES]
    if unknown:
        ap.error(f"unknown suite(s) {unknown}; choose from {sorted(SUITES)}")
    if a.out and len(suites) > 1:
        ap.error("--out needs a single suite")
    todo = []
    for suite in suites:
        out = a.out or f"runs/{suite}"
        if not a.dry:
            os.makedirs(os.path.join(out, "_logs"), exist_ok=True)
        jobs = [(n, c, out) for n, c in jobs_for(suite, out, a.extra, a.data_dir)
                if not os.path.exists(os.path.join(out, n, "results.json"))]
        if a.only:
            keep = set(a.only.split(","))
            jobs = [j for j in jobs if j[1][j[1].index("--variant") + 1] in keep]
        reused = sorted({os.path.dirname(os.path.dirname(p)) for p in reused_runs(suite, out).values()
                         if os.path.exists(p)})
        print(f"{len(jobs)} jobs to run in {out}" + (f" (identical runs in {', '.join(reused)} are reused, "
                                                    f"not retrained)" if reused else ""))
        todo += jobs
    if a.dry:
        for n, c, _ in todo:
            print(" ".join(c))
        return
    run_jobs(todo, a.gpus, a.per_gpu, a.retries)

if __name__ == "__main__":
    main()
