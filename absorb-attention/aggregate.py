import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except ImportError:
    sps = None
try:
    from sweep import SUITES, reused_runs
except ImportError:
    SUITES, reused_runs = {}, None

ZS_MAIN = {
    "lambada_openai": "acc", "hellaswag": "acc_norm", "piqa": "acc_norm", "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm", "winogrande": "acc", "sciq": "acc_norm", "openbookqa": "acc_norm",
}


def long_context(run_dir):
    path = os.path.join(run_dir, "long_context.json")
    if not os.path.exists(path):
        return {}, []
    with open(path) as fh:
        z = json.load(fh)
    cols = {f"lc_{k}": v for k, v in z.get("ppl", {}).items() if k.startswith("ppl@")}
    cols.update({k: v for k, v in z.get("passkey", {}).items() if k.startswith("passkey@") and "depth" not in k})
    return cols, (z.get("evict") or []) + (z.get("evict_nosink") or [])


def zeroshot_cols(run_dir):
    path = os.path.join(run_dir, "zeroshot.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        z = json.load(fh)
    if z.get("limit") is not None:
        return {}
    out = {}
    for task, metrics in z.get("tasks", {}).items():
        key = ZS_MAIN.get(task, "acc_norm" if "acc_norm" in metrics else "acc")
        if key in metrics:
            out[f"zs_{task}"] = metrics[key]
        if task == "lambada_openai" and "perplexity" in metrics:
            out["zs_lambada_ppl"] = metrics["perplexity"]
    accs = [v for k, v in out.items() if k.startswith("zs_") and k != "zs_lambada_ppl"]
    if len(accs) == len(ZS_MAIN):
        out["zs_avg"] = sum(accs) / len(accs)
    return out


STAT_KEYS = ["null_mean", "beta_mean", "first_mass", "punct_mass", "max_act", "R_mean", "retired_frac_R>2",
             "R_bos_mean"]


def mode_of(r):
    args = r.get("args") or {}
    comp = r.get("compile") or ("on" if args.get("compile") else "off")
    ckpt = bool(args.get("grad_ckpt")) or "grad_ckpt" in (r.get("oom_fix") or [])
    return ("compiled" if comp == "on" else "eager") + ("+ckpt" if ckpt else "")


def reused(d):
    suite = os.path.basename(os.path.normpath(d))
    if reused_runs is None or "reuse" not in SUITES.get(suite, {}):
        return []
    found, missing = [], []
    for n, src in reused_runs(suite, d).items():
        if not os.path.exists(os.path.join(d, n, "results.json")):
            (found if os.path.exists(src) else missing).append((src, n))
    if missing:
        print(f"[aggregate] {suite}: {len(missing)} runs it takes from "
              f"{', '.join(sorted({os.path.basename(os.path.dirname(os.path.dirname(s))) for s, _ in missing}))} "
              f"are not there: put that suite's run dirs next to {d}", file=sys.stderr)
    return found


def mark_reruns(df):
    if df.empty:
        return df
    key = ["task", "variant", "seed"]
    df["calib"] = df.duplicated(key, keep="first")
    first = df[~df["calib"]].set_index(key)
    src = first["suite"] + " (" + first["mode"] + ")"
    df["rerun_of"] = [src.get(k, "") if c else "" for k, c in zip(map(tuple, df[key].values), df["calib"])]
    return df


def load(dirs):
    rows, layers, curves, evicts = [], [], [], []
    for d in dirs:
        files = [(f, None) for f in sorted(glob.glob(os.path.join(d, "*", "results.json")))] + reused(d)
        for f, as_name in files:
            with open(f) as fh:
                r = json.load(fh)
            suffix = f"_{r['variant']}_s{r['seed']}"
            name = as_name or r["name"]
            tcfg = name[:-len(suffix)] if name.endswith(suffix) else r["task"]
            rid = len(rows)
            src = os.path.relpath(os.path.dirname(f), os.path.dirname(os.path.normpath(d)) or ".") if as_name else ""
            row = {"suite": os.path.basename(os.path.normpath(d)), "task": tcfg, "variant": r["variant"],
                   "seed": r["seed"], "params_M": r["params_nonemb"] / 1e6, "steps": r["steps_done"],
                   "tok_per_s": r.get("train_tok_per_s"), "peak_MiB": r.get("train_peak_MiB"),
                   "nonfinite": r.get("nonfinite_grad_steps", 0),
                   "gpu": r.get("gpu", "?"), "amp": r.get("amp_dtype", "?"), "mode": mode_of(r), "run": rid,
                   "reused_from": src.replace(os.sep, "/")}
            row.update({k: v for k, v in r["final"].items() if v is not None})
            row.update(zeroshot_cols(os.path.dirname(f)))
            lc_cols, lc_evict = long_context(os.path.dirname(f))
            row.update(lc_cols)
            if r["task"] != "lm":
                hits = [h["step"] for h in r.get("history", []) if (h.get("acc@1x") or 0) >= 0.9]
                if not hits and r["final"].get("acc@1x", 0) >= 0.9:
                    hits = [r["steps_done"]]
                row["steps_to_90"] = hits[0] if hits else np.inf
            row.update({f"st_{k}": v for k, v in r["stats"]["summary"].items()})
            row.update({f"bench_{k}": v for k, v in (r.get("bench") or {}).items()})
            rows.append(row)
            for e in ([] if as_name else (r.get("evict") or []) + lc_evict):
                evicts.append({"task": tcfg, "variant": r["variant"], "seed": r["seed"], "run": rid,
                               "setting": e["policy"] + (f"@{e['budget']}" if "budget" in e else "")
                               + (f"@tau{e['tau']}" if "tau" in e else "")
                               + (f"@sink{e['n_sink']}" if e.get("n_sink", 4) != 4 else ""),
                               "acc": e["acc"], "ppl": e.get("ppl"), "kv_frac": e["kv_frac_mean"],
                               "kv_final": e["kv_final_frac"]})
            for h in r.get("history", []):
                curves.append({"task": tcfg, "variant": r["variant"], "seed": r["seed"], "run": rid,
                               "step": h["step"], "acc": h.get("acc@1x"), "ppl": h.get("ppl@1x")})
            for li, l in enumerate(r["stats"]["layers"]):
                layers.append({"task": tcfg, "variant": r["variant"], "seed": r["seed"], "run": rid, "layer": li,
                               **{k: v for k, v in l.items() if not isinstance(v, list)}})
    return mark_reruns(pd.DataFrame(rows)), pd.DataFrame(layers), pd.DataFrame(curves), pd.DataFrame(evicts)


def main_runs(df, *frames):
    if "calib" not in df:
        return (df,) + frames
    keep = set(df.loc[~df["calib"], "run"])
    return (df[~df["calib"]],) + tuple(x[x["run"].isin(keep)] if "run" in x else x for x in frames)


def main_metrics(task):
    if task == "lm":
        return ["ppl@1x", "ppl@2x", "ppl@4x"], False
    return ["acc@1x", "acc@2x", "acc@4x"], True


def fmt(m, s, n, pct):
    if np.isnan(m):
        return "–"
    if pct:
        return f"{100 * m:.1f} ± {100 * s:.1f}" if n > 1 else f"{100 * m:.1f}"
    return f"{m:.2f} ± {s:.2f}" if n > 1 else f"{m:.2f}"


def paired_p(df, task, var, ref, metric):
    if sps is None:
        return np.nan
    df = main_runs(df)[0]

    def per_seed(v):
        s = df[(df.task == task) & (df.variant == v)].drop_duplicates("seed")
        return s.set_index("seed")[metric]

    a, b = per_seed(var), per_seed(ref)
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    x, y = a.loc[common].values, b.loc[common].values
    if np.allclose(x, y):
        return 1.0
    return float(sps.ttest_rel(x, y).pvalue)


def tok_s(s, by_mode):
    if not s["tok_per_s"].notna().any():
        return "–"
    if not by_mode:
        return f"{s['tok_per_s'].mean():.0f}"
    return "; ".join(f"{m:.0f} {md}" for md, m in s.groupby("mode")["tok_per_s"].mean().dropna().items())


def calib_table(full, df):
    if "calib" not in full or not full["calib"].any():
        return []
    lines = ["\n### Reruns of an earlier (task, variant, seed): calibration of the training protocol "
             "(not in the means and tests above)\n",
             "| task / variant / seed | metric | first run | rerun | rerun − first | tok/s first → rerun |",
             "|---|---|---|---|---|---|"]
    first = df.set_index(["task", "variant", "seed"])
    for _, r in full[full["calib"]].iterrows():
        m = main_metrics(r.task)[0][0]
        f = first.loc[(r.task, r.variant, r.seed)]
        a, b = f.get(m, np.nan), r.get(m, np.nan)
        note = "" if f["mode"] != r["mode"] else " (same mode: duplicate)"
        if f["steps"] != r["steps"] or abs(f["params_M"] - r["params_M"]) > 1e-6:
            note += f" (other budget/size: {f['steps']} vs {r['steps']} steps)"
        lines.append(f"| {r.task} / {r.variant} / s{r.seed} | {m} | {a:.4f} {f['suite']} ({f['mode']}) | "
                     f"{b:.4f} {r.suite} ({r['mode']}){note} | {b - a:+.4f} | "
                     f"{f['tok_per_s'] or np.nan:.0f} → {r['tok_per_s'] or np.nan:.0f} |")
    return lines


def tables(df, out):
    full, df = df, main_runs(df)[0]
    lines = []
    for task in sorted(df.task.unique()):
        metrics, pct = main_metrics(task)
        sub = df[df.task == task]
        g = sub.groupby("variant")
        lines.append(f"\n### Task: {task}\n")
        mode = sub["mode"] if "mode" in sub else pd.Series("?", index=sub.index)
        hw = sub.assign(mode=mode).groupby(["gpu", "amp", "mode"]).size()
        by_mode = mode.nunique() > 1
        if len(hw) > 1:
            lines.append("**Warning: mixed hardware/precision/training mode in this table:** "
                         + ", ".join(f"{gp} {am} {md} ({n} runs)" for (gp, am, md), n in hw.items())
                         + (" — tok/s is given per mode" if by_mode else "") + "\n")
        else:
            lines.append(f"hardware: {hw.index[0][0]} ({hw.index[0][1]}, {hw.index[0][2]})\n")
        reuse = sub["reused_from"][sub["reused_from"] != ""] if "reused_from" in sub else []
        if len(reuse):
            lines.append(f"{len(reuse)} runs are the identical runs of "
                         f"{', '.join(sorted({x.split('/')[0] for x in reuse}))} (sweep.py reuse, not retrained)\n")
        stat_cols = [f"st_{k}" for k in STAT_KEYS if f"st_{k}" in sub]
        extra = ["steps_to_90"] if "steps_to_90" in sub else []
        hdr = (["variant", "n"] + metrics + ["p vs vanilla (" + metrics[0] + ")"] + extra + [c[3:] for c in stat_cols]
               + ["tok/s", "params M"])
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        order = g[metrics[0]].mean().sort_values(ascending=not pct).index
        for v in order:
            s = g.get_group(v)
            cells = [v, str(len(s))]
            for m in metrics:
                cells.append(fmt(s[m].mean(), s[m].std(ddof=1) if len(s) > 1 else 0, len(s), pct) if m in s else "–")
            p = paired_p(df, task, v, "vanilla", metrics[0]) if v != "vanilla" and metrics[0] in s else np.nan
            cells.append("–" if np.isnan(p) else f"{p:.3f}")
            if extra:
                st = s["steps_to_90"]
                cells.append(f"{np.median(st):.0f} ({int(np.isfinite(st).sum())}/{len(st)})")
            for c in stat_cols:
                cells.append(f"{s[c].mean():.3g}" if c in s and s[c].notna().any() else "–")
            cells.append(tok_s(s.assign(mode=mode.loc[s.index]), by_mode))
            cells.append(f"{s['params_M'].mean():.2f}")
            lines.append("| " + " | ".join(cells) + " |")
    zs = [c for c in df.columns if c.startswith("zs_")]
    if zs:
        order = [f"zs_{t}" for t in ZS_MAIN if f"zs_{t}" in zs] + [c for c in ("zs_avg", "zs_lambada_ppl") if c in zs]
        lines.append("\n### Zero-shot (lm-evaluation-harness, mean ± sd over seeds; acc_norm where available)\n")
        lines.append("| task / variant | n | " + " | ".join(c[3:] for c in order) + " |")
        lines.append("|" + "---|" * (len(order) + 2))
        for (task, v), s in df[df[order[0]].notna()].groupby(["task", "variant"]):
            cells = [f"{task} / {v}", str(len(s))]
            for c in order:
                m, sd = s[c].mean(), s[c].std(ddof=1)
                pct = c != "zs_lambada_ppl"
                cells.append(fmt(m, 0 if np.isnan(sd) else sd, len(s), pct))
            lines.append("| " + " | ".join(cells) + " |")
    lc = sorted((c for c in df.columns if c.startswith("lc_ppl@")), key=lambda c: int(c.split("@")[1][:-1]))
    pk = sorted((c for c in df.columns if c.startswith("passkey@")), key=lambda c: int(c.split("@")[1]))
    if lc or pk:
        lines.append("\n### Long context (perplexity at k x the training length; passkey accuracy by length)\n")
        hdr = [c[3:] for c in lc] + pk
        lines.append("| task / variant | n | " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * (len(hdr) + 2))
        for (task, v), s in df[df[(lc or pk)[0]].notna()].groupby(["task", "variant"]):
            cells = [f"{task} / {v}", str(len(s))]
            for c in lc:
                cells.append(fmt(s[c].mean(), s[c].std(ddof=1) if len(s) > 1 else 0, len(s), False))
            for c in pk:
                cells.append(fmt(s[c].mean(), s[c].std(ddof=1) if len(s) > 1 else 0, len(s), True))
            lines.append("| " + " | ".join(cells) + " |")
    lines += calib_table(full, df)
    md = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


def per_layer(layers, out):
    if layers.empty:
        return
    keys = [k for k in ["null_mean", "beta_mean", "first_mass", "R_mean", "R_bos_mean", "max_act"] if k in layers]
    t = layers.groupby(["task", "variant", "layer"])[keys].mean().round(4)
    t.to_csv(os.path.join(out, "per_layer.csv"))
    with open(os.path.join(out, "per_layer.md"), "w", encoding="utf-8") as f:
        f.write(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())


def evict_table(ev, out):
    if ev.empty:
        return
    g = ev.groupby(["task", "variant", "setting"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                                        ppl=("ppl", "mean"), kv_frac=("kv_frac", "mean"),
                                                        n=("seed", "count")).round(4)
    g.to_csv(os.path.join(out, "eviction.csv"))
    with open(os.path.join(out, "eviction.md"), "w", encoding="utf-8") as f:
        f.write(g.to_markdown() if hasattr(g, "to_markdown") else g.to_string())


def curve_plots(curves, out):
    if curves.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task, sub in curves.groupby("task"):
        key = "ppl" if task == "lm" else "acc"
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for v, s in sub.groupby("variant"):
            g = s.groupby("step")[key]
            m, lo, hi = g.mean(), g.min(), g.max()
            line, = ax.plot(m.index, m.values, label=v, lw=2.2 if v == "absorb" else 1.2)
            ax.fill_between(m.index, lo.values, hi.values, color=line.get_color(), alpha=0.12)
        ax.set_xlabel("training step")
        ax.set_ylabel("answer accuracy (mean, min–max over seeds)" if key == "acc" else "val perplexity")
        ax.set_title(task)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"curve_{task}.png"), dpi=150)
        plt.close(fig)
        if key != "acc":
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for v, s in sub.groupby("variant"):
            solved = s.assign(ok=s["acc"] >= 0.9).sort_values("step")
            solved["ok"] = solved.groupby("seed")["ok"].cummax()
            frac = solved.groupby("step")["ok"].mean()
            ax.step(frac.index, frac.values, where="post", label=f"{v} (n={s.seed.nunique()})",
                    lw=2.2 if v == "absorb" else 1.2)
        ax.set_xlabel("training step")
        ax.set_ylabel("fraction of seeds with acc ≥ 90 %")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(task)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"solved_{task}.png"), dpi=150)
        plt.close(fig)


def plots(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task in sorted(df.task.unique()):
        metrics, pct = main_metrics(task)
        sub = df[df.task == task]
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [int(m.split("@")[1][:-1]) for m in metrics]
        for v, s in sub.groupby("variant"):
            ys = [s[m].mean() if m in s else np.nan for m in metrics]
            ax.plot(xs, ys, marker="o", label=v, lw=2.2 if v.startswith("absorb") and v == "absorb" else 1.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("eval length / train length")
        ax.set_ylabel("answer accuracy" if pct else "perplexity")
        ax.set_title(task)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"extrap_{task}.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, layers, curves, ev = load(a.dirs)
    if df.empty:
        print("no results found")
        return
    print(tables(df, a.out))
    main_df, layers, curves, ev = main_runs(df, layers, curves, ev)
    main_df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    df[df["calib"]].to_csv(os.path.join(a.out, "reruns.csv"), index=False)
    per_layer(layers, a.out)
    plots(main_df, a.out)
    curve_plots(curves, a.out)
    evict_table(ev, a.out)


if __name__ == "__main__":
    main()
