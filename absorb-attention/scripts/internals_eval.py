import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from attention.common import Rotary, up
from model import Ctx

BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
POS_BUCKETS = [(0, 1), (1, 8), (8, 64), (64, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192),
               (8192, 16384)]
FAR = 64
_STREAMED = False


def load_json(path, log=print):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        log(f"ignoring unreadable {path}: {type(e).__name__}: {e}")
        return {}
    return r if isinstance(r, dict) else {}


def save_json(obj, path, indent=1):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def autocast(dtype):
    if dtype is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def hidden(model, idx, chunk=0):
    ctx = Ctx(chunk=chunk)
    x = model.emb(idx)
    for blk in model.blocks:
        x = blk(x, ctx)
    return model.norm(x)


def token_nll(model, h, tgt, step=2048):
    out = []
    for p in range(0, h.shape[1], step):
        lg = up(model.head(h[:, p:p + step]))
        out.append(F.cross_entropy(lg.flatten(0, 1), tgt[:, p:p + step].flatten(), reduction="none")
                   .view(tgt.shape[0], -1))
    return torch.cat(out, dim=1)


@torch.no_grad()
def full_nll(model, inp, tgt, dtype, chunk=512):
    with autocast(dtype):
        h = hidden(model, inp, chunk)
    return token_nll(model, h, tgt)


@torch.no_grad()
def window_nll(model, inp, tgt, window, block, n_sink, dtype, chunk=512, bs=16):
    B, N = inp.shape
    W = min(window, N)
    odt = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32
    out = torch.empty(B, N, dtype=odt, device=inp.device)
    out[:, :W] = full_nll(model, inp[:, :W], tgt[:, :W], dtype, chunk)
    if N == W:
        return out
    block = max(1, min(block, W - n_sink))
    spans = [(p, min(N, p + block)) for p in range(W, N, block)]
    ctxs = [torch.cat([inp[:, :n_sink], inp[:, e - (W - n_sink):e]], dim=1) for _, e in spans]
    for g in range(0, len(spans), max(1, bs // B)):
        grp = list(range(g, min(len(spans), g + max(1, bs // B))))
        X = torch.cat([ctxs[i] for i in grp], dim=0)
        with autocast(dtype):
            h = hidden(model, X, chunk)
        for k, i in enumerate(grp):
            p, e = spans[i]
            hk = h[k * B:(k + 1) * B, W - (e - p):]
            out[:, p:e] = token_nll(model, hk, tgt[:, p:e])
    return out


def val_windows(data_dir, train_len, scale, budget, device):
    from tasks import LMData
    data = LMData(data_dir, seq_len=train_len)
    n_tok = train_len * scale
    n_win = max(1, budget // n_tok)
    b = max(1, min(8, n_win, 8192 // n_tok if n_tok <= 8192 else 1))
    return list(data.fixed_batches(b, math.ceil(n_win / b), scale=scale, device=device))


def load_longdocs(data_dir, cache_dir, n_docs, length, log, sources=("emozilla/pg19", "deepmind/pg19")):
    global _STREAMED
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"longdocs_{length}.npz")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh, np.load(fh, allow_pickle=False) as z:
                return z["tokens"][:n_docs], str(z["source"])
        except Exception as e:
            log(f"[longdoc] rebuilding unreadable {path}: {type(e).__name__}: {str(e)[:200]}")
    docs, source, tok = [], None, None
    for name in sources:
        kw = {"trust_remote_code": True} if name == "deepmind/pg19" else {}
        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
            tok = tok or GPT2TokenizerFast.from_pretrained("gpt2", token=os.environ.get("HF_TOKEN") or None)
            _STREAMED = True
            ds = load_dataset(name, split="test", streaming=True, token=os.environ.get("HF_TOKEN") or None, **kw)
            for ex in ds:
                ids = tok.encode(ex["text"])
                if len(ids) >= length:
                    docs.append([tok.eos_token_id] + ids[:length])
                if len(docs) >= n_docs:
                    break
            if docs:
                source = f"{name}:test"
                break
        except Exception as e:
            log(f"[longdoc] {name} unavailable: {type(e).__name__}: {str(e)[:200]}")
            docs = []
    if not docs:
        with open(os.path.join(data_dir, "meta.json")) as f:
            eot = json.load(f).get("eot", 50256)
        val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ends = np.flatnonzero(val == eot)
        start = 0
        for e in ends:
            if e - start >= length:
                docs.append([eot] + val[start:start + length].astype(np.int64).tolist())
            start = e + 1
            if len(docs) >= n_docs:
                break
        source = "fineweb-edu:val(>=%d tokens)" % length
    if not docs:
        raise RuntimeError(f"no document of >= {length} tokens in {list(sources)} or {data_dir}/val.bin")
    arr = np.asarray(docs, dtype=np.int64)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, tokens=arr, source=source)
    os.replace(tmp, path)
    return arr[:n_docs], source


def run_window(model, a, device, dtype, log):
    out = {}
    for s in [int(x) for x in a.scales.split(",")]:
        tot = {"full": 0.0, "sw": 0.0, "stream": 0.0}
        cnt = 0
        for d in val_windows(a.data_dir, a.train_len, s, a.budget, device):
            inp, tgt = d["input"], d["target"]
            tot["full"] += float(full_nll(model, inp, tgt, dtype).sum())
            tot["sw"] += float(window_nll(model, inp, tgt, a.train_len, a.block, 0, dtype).sum())
            tot["stream"] += float(window_nll(model, inp, tgt, a.train_len, a.block, a.n_sink, dtype).sum())
            cnt += tgt.numel()
        for k, v in tot.items():
            out[f"{k}_ppl@{s}x"] = math.exp(v / cnt)
        out[f"tokens@{s}x"] = cnt
        log(f"[window] {s}x full {out[f'full_ppl@{s}x']:.3f} sliding {out[f'sw_ppl@{s}x']:.3f} "
            f"streaming {out[f'stream_ppl@{s}x']:.3f}")
    out.update({"window": a.train_len, "block": a.block, "n_sink": a.n_sink})
    return out


def run_longdoc(model, a, device, dtype, log):
    srcs = tuple(x for x in a.longdoc_sources.split(",") if x and x != "fineweb")
    toks, source = load_longdocs(a.data_dir, a.cache_dir, a.longdoc_n, a.longdoc_len, log, srcs)
    L = toks.shape[1] - 1
    sums = {"full": np.zeros(L), "sw": np.zeros(L)}
    for i in range(len(toks)):
        x = torch.from_numpy(toks[i:i + 1]).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        sums["full"] += full_nll(model, inp, tgt, dtype)[0].double().cpu().numpy()
        sums["sw"] += window_nll(model, inp, tgt, a.train_len, a.block, 0, dtype)[0].double().cpu().numpy()
    n = len(toks)
    out = {"source": source, "n_docs": n, "length": L}
    for k, v in sums.items():
        nll = v / n
        out[k] = {f"{lo}-{hi}": float(math.exp(nll[lo:min(hi, L)].mean())) for lo, hi in BUCKETS if lo < L}
        out[k + "_curve"] = [float(nll[p:p + 256].mean()) for p in range(0, L, 256)]
    log(f"[longdoc] {source} x{n}: full {out['full']} sliding {out['sw']}")
    return out


ROPE_METHODS = ("ntk", "pi")


@contextlib.contextmanager
def rope_scaled(model, factor, method):
    rots = [m for m in model.modules() if isinstance(m, Rotary)]
    saved = [r.inv_freq for r in rots]
    try:
        for r, inv in zip(rots, saved):
            d = 2 * inv.numel()
            if method == "ntk":
                ex = torch.arange(0, d, 2, dtype=torch.float64, device=inv.device) / max(1, d - 2)
                new = inv.double() * torch.pow(float(factor), -ex)
            elif method == "pi":
                new = inv.double() / float(factor)
            else:
                raise ValueError(method)
            r.inv_freq = new.to(inv.dtype)
        yield len(rots)
    finally:
        for r, inv in zip(rots, saved):
            r.inv_freq = inv


@torch.no_grad()
def run_rope(model, a, device, dtype, log):
    if not any(isinstance(m, Rotary) for m in model.modules()):
        log("[rope] no RoPE layers (NoPE model): nothing to scale")
        return {"rope": False}
    out = {"rope": True, "methods": list(ROPE_METHODS)}
    for s in [int(x) for x in a.scales.split(",")]:
        tot = dict.fromkeys(("full",) + ROPE_METHODS, 0.0)
        cnt = 0
        for d in val_windows(a.data_dir, a.train_len, s, a.budget, device):
            inp, tgt = d["input"], d["target"]
            tot["full"] += float(full_nll(model, inp, tgt, dtype).sum())
            for m in ROPE_METHODS:
                with rope_scaled(model, s, m):
                    tot[m] += float(full_nll(model, inp, tgt, dtype).sum())
            cnt += tgt.numel()
        for k, v in tot.items():
            out[f"{k}_ppl@{s}x"] = math.exp(v / cnt)
        out[f"tokens@{s}x"] = cnt
        log(f"[rope] {s}x full {out[f'full_ppl@{s}x']:.3f} " + " ".join(f"{m} {out[f'{m}_ppl@{s}x']:.3f}"
                                                                     for m in ROPE_METHODS))
    return out


def _setting_label(r):
    b = r.get("budget")
    lab = r["policy"] + (f" tau={r['tau']:g}" if r.get("tau") is not None else "")
    if b is not None:
        lab += " budget=" + ("per-layer" if isinstance(b, list) else str(b))
    return lab + (f" n_sink={r.get('n_sink', 4)}" if r["policy"] in ("window", "stream") else "")


DECODE_KEYS = ("policy", "tau", "budget", "n_sink")


def _same_setting(r, st):
    return all(r.get(k, d) == st.get(k, d) for k, d in
               (("policy", None), ("tau", None), ("budget", None), ("n_sink", 4), ("alloc", None), ("match", None),
                ("windows", None)))


def _eval_settings(model, settings, batches, device, dtype, done, cur, save):
    from inference import eval_eviction
    for st in settings:
        i = len(cur)
        if not (i < len(done) and _same_setting(done[i], st)):
            del done[i:]
            r = eval_eviction(model, None, [{k: v for k, v in st.items() if k in DECODE_KEYS}],
                              device=str(device), amp_dtype=dtype, batches=batches)[0]
            r.update({k: v for k, v in st.items() if k not in DECODE_KEYS})
            done.append(r)
            save()
        cur.append(done[i])


def _evict_scales(a):
    return [(int(x.split(":")[0]), *(int(y) for y in x.split(":")[1].split("x"))) for x in a.evict_scales.split(",")]


def _retiring(model):
    from attention import AbsorbAttention, SelectiveAttention
    att = model.blocks[0].attn
    return isinstance(att, SelectiveAttention) or (isinstance(att, AbsorbAttention) and model.cfg.retire)


def layer_alloc(kv_final_per_layer, mean_budget, N):
    w = np.asarray(kv_final_per_layer) + 1e-6
    return [int(max(16, min(N, round(mean_budget * x / w.mean())))) for x in w]


def matched_budget(t4, n_sink, N):
    m = float(np.mean(t4["kv_peak_per_layer"]))
    return int(max(n_sink + 1, min(N, round(m)))), m


def heldout_batches(data, B, nb, s, device):
    held = list(data.fixed_batches(B, 2 * nb, scale=s, device=device))[nb:]
    if len(held) != nb or any(d["input"].shape[0] != B for d in held):
        raise RuntimeError(f"the validation set has fewer than {2 * B * nb} windows of {data.seq_len * s + 1} tokens")
    return held


def window_starts(N, lo, hi):
    return [i * (N + 1) for i in range(lo, hi)]


def run_evict(model, a, device, dtype, log, res=None, save=None):
    from tasks import LMData
    res = {} if res is None else res
    save = save or (lambda: None)
    retiring = _retiring(model)
    data = LMData(a.data_dir, seq_len=a.train_len)
    part = res.setdefault("evict_partial", {})
    out = {}
    for s, B, nb in _evict_scales(a):
        key = f"{s}x"
        if key in part:
            out[key] = part[key]
            log(f"[evict {key}] kept from an earlier run ({len(part[key])} settings)")
            continue
        N = a.train_len * s
        batches = list(data.fixed_batches(B, nb, scale=s, device=device))
        run = res.get("evict_running") or {}
        done = list(run.get("results", [])) if run.get("scale") == key and run.get("batches") == [B, nb] else []
        res["evict_running"] = {"scale": key, "batches": [B, nb], "results": done}
        cur = []

        def ev(settings, done=done, cur=cur, batches=batches):
            _eval_settings(model, settings, batches, device, dtype, done, cur, save)

        b1, b2 = N // 4, N // 8
        if retiring:
            ev([{"policy": "none"}] + [{"policy": "retire_tau", "tau": t} for t in (2.0, 4.0, 8.0)])
        else:
            ev([{"policy": "none"}])
        rest = []
        for b in (b1, b2):
            rest += [{"policy": "tova", "budget": b}, {"policy": "tova_layer", "budget": b}]
            rest += [{"policy": "window", "budget": b, "n_sink": ns} for ns in dict.fromkeys((a.n_sink, 0))]
            rest.append({"policy": "stream", "budget": b, "n_sink": a.n_sink})
            if retiring:
                rest.append({"policy": "retire_budget", "budget": b})
        if retiring:
            t4 = next(r for r in cur if r.get("tau") == 4.0)
            rest.append({"policy": "retire_budget", "budget": layer_alloc(t4["kv_final_per_layer"], b2, N),
                         "alloc": "tau4"})
        if s > 4:
            rest = [r for r in rest if (r.get("budget") == b2 and r.get("n_sink", a.n_sink) == a.n_sink)
                    or r.get("alloc")]
            for st in [{"policy": "stream", "budget": a.train_len, "n_sink": ns} for ns in (a.n_sink,) if ns > 0] \
                    + [{"policy": "window", "budget": a.train_len, "n_sink": 0}]:
                if st not in rest:
                    rest.append(st)
        ev(rest)
        base = cur[0]["ppl"]
        for r in cur:
            r["dppl_pct"] = 100 * (r["ppl"] / base - 1)
            log(f"[evict {key}] {_setting_label(r)} ppl={r['ppl']:.3f} ({r['dppl_pct']:+.2f}%) "
                f"kv={r['kv_frac_mean']:.3f} final={r['kv_final_frac']:.3f} peak={r['kv_peak_keys']:.0f} "
                f"evicted_mass={r['evicted_mass_mean']:.4f}")
        part[key] = out[key] = cur
        res.pop("evict_running", None)
        save()
    model.eval()
    return out


MATCH = "tau4_peak_layer"
HELDOUT = "tau4_heldout"


def run_evict_matched(model, a, device, dtype, log, res=None, save=None):
    from tasks import LMData
    res = {} if res is None else res
    save = save or (lambda: None)
    if not _retiring(model):
        log("[evict_matched] no retirement penalty: nothing to add")
        return {}
    ev_old = res.get("evict")
    if not isinstance(ev_old, dict):
        raise RuntimeError("evict_matched needs the finished 'evict' part (run --what evict,evict_matched)")
    data = LMData(a.data_dir, seq_len=a.train_len)
    part = res.setdefault("evict_matched_partial", {})
    out = {}
    for s, B, nb in _evict_scales(a):
        key = f"{s}x"
        if key in part:
            out[key] = part[key]
            log(f"[evict_matched {key}] kept from an earlier run ({len(part[key])} rows)")
            continue
        N = a.train_len * s
        rows = ev_old.get(key)
        if not rows:
            raise RuntimeError(f"'evict' has no {key} rows (scales {sorted(ev_old)}; --evict_scales {a.evict_scales})")
        none = next((r for r in rows if r["policy"] == "none"), None)
        t4 = next((r for r in rows if r["policy"] == "retire_tau" and r.get("tau") == 4.0), None)
        if none is None or t4 is None:
            raise RuntimeError(f"'evict' {key} has no full-context or tau = 4 row to reuse")
        for r in (none, t4):
            if r.get("tokens", B * nb * N) != B * nb * N or r.get("seq_len", N) != N:
                raise RuntimeError(f"'evict' {key} was decoded on other sequences ({r.get('tokens')} tokens of "
                                   f"{r.get('seq_len')}) than --evict_scales {s}:{B}x{nb} ({B * nb} x {N})")
        batches = list(data.fixed_batches(B, nb, scale=s, device=device))
        held = heldout_batches(data, B, nb, s, device)
        run = res.get("evict_matched_running") or {}
        done = list(run.get("results", [])) if run.get("scale") == key and run.get("batches") == [B, nb] else []
        res["evict_matched_running"] = {"scale": key, "batches": [B, nb], "results": done}
        cur = []
        _eval_settings(model, [{"policy": "retire_tau", "tau": 4.0, "windows": "heldout"}], held, device, dtype,
                       done, cur, save)
        calib = dict(cur[0], starts=window_starts(N, B * nb, 2 * B * nb), eval_starts=window_starts(N, 0, B * nb))
        bm, peak = matched_budget(t4, a.n_sink, N)
        new = [{"policy": p, "budget": bm, "n_sink": a.n_sink, "match": MATCH, "match_peak": peak}
               for p in ("window", "stream")]
        new.append({"policy": "retire_budget", "budget": layer_alloc(calib["kv_final_per_layer"], N // 8, N),
                    "alloc": HELDOUT, "calib": calib})
        _eval_settings(model, new, batches, device, dtype, done, cur, save)
        lst = [dict(none, reused="evict")] + cur[1:]
        for r in lst[1:]:
            r["dppl_pct"] = 100 * (r["ppl"] / none["ppl"] - 1)
        for r in lst:
            log(f"[evict_matched {key}] {_setting_label(r)} [{r.get('match') or r.get('alloc') or 'reused'}] "
                f"ppl={r['ppl']:.3f} ({r.get('dppl_pct', 0.0):+.2f}%) kv={r['kv_frac_mean']:.3f} "
                f"peak={r['kv_peak_keys']:.0f}")
        log(f"[evict_matched {key}] matched budget {bm} (tau=4 per-layer peak {peak:.1f}); held-out tau=4 final per layer "
            f"{[round(x, 3) for x in calib['kv_final_per_layer']]} (evaluation sequences: "
            f"{[round(x, 3) for x in t4['kv_final_per_layer']]})")
        part[key] = out[key] = lst
        res.pop("evict_matched_running", None)
        save()
    model.eval()
    return out


def _bucket_means(v, buckets):
    N = v.shape[0]
    return {f"{lo}-{hi}": float(v[lo:min(hi, N)].mean()) for lo, hi in buckets if lo < N}


@torch.no_grad()
def run_stats(model, a, device, dtype, log):
    from attention import AbsorbAttention, SinkAttention
    from tasks import LMData
    data = LMData(a.data_dir, seq_len=a.train_len)
    L = len(model.blocks)
    acc = [dict() for _ in range(L)]
    batches = list(data.fixed_batches(4, a.stats_batches, device=device))

    def add(li, k, v):
        acc[li].setdefault(k, []).append(v)

    hooks = []
    cur = {}

    def attn_hook(li):
        def fn(mod, inputs, output):
            h = inputs[0]
            aux = output[1]
            a_ = aux.get("attn")
            n = aux.get("null")
            sp = cur["special"]
            if a_ is not None:
                a_ = a_.float()
                spm = sp.to(a_.dtype)[:, None, None, :]
                variants = [("", a_)]
                if n is not None:
                    keys = a_.sum(-1).clamp_min(1e-12)
                    variants.append(("_renorm", a_ / keys.unsqueeze(-1)))
                for suf, w in variants:
                    fm, pm = w[:, :, :, 0], (w * spm).sum(-1)
                    add(li, "first_mass" + suf, float(fm[:, :, 1:].mean()))
                    add(li, "punct_mass" + suf, float(pm[:, :, 1:].mean()))
                    add(li, "first_mass" + suf + "_by_pos", _bucket_means(fm.mean(dim=(0, 1)), POS_BUCKETS))
                    add(li, "punct_mass" + suf + "_by_pos", _bucket_means(pm.mean(dim=(0, 1)), POS_BUCKETS))
                    if fm.shape[-1] > FAR:
                        add(li, f"first_mass{suf}_ge{FAR}", float(fm[:, :, FAR:].mean()))
                        add(li, f"punct_mass{suf}_ge{FAR}", float(pm[:, :, FAR:].mean()))
            if n is not None:
                n = n.float()
                add(li, "null_mean", float(n.mean()))
                add(li, "null_by_pos", _bucket_means(n.mean(dim=(0, 1)), POS_BUCKETS))
                s = None
                if isinstance(mod, AbsorbAttention):
                    s = mod.null_logits(h)
                elif isinstance(mod, SinkAttention):
                    s = mod.sink.float().view(1, -1, 1)
                if s is not None:
                    s = s.float().expand_as(n)
                    if a_ is not None:
                        tiny = torch.finfo(torch.float32).tiny
                        gate = torch.log(a_.sum(-1).clamp_min(tiny)) - torch.log(n.clamp_min(tiny))
                    else:
                        gate = torch.log1p(-n.clamp(max=1 - 1e-7)) - torch.log(n.clamp_min(1e-30))
                    lam = gate + s
                    vl, vs = lam.var(dim=(0, 2)), s.var(dim=(0, 2))
                    cov = ((lam - lam.mean(dim=(0, 2), keepdim=True)) * (s - s.mean(dim=(0, 2), keepdim=True))
                           ).mean(dim=(0, 2))
                    add(li, "gate_var", float(gate.var(dim=(0, 2)).mean()))
                    add(li, "gate_var_lambda", float(vl.mean()))
                    add(li, "gate_var_s", float(vs.mean()))
                    add(li, "gate_cov_lambda_s", float(cov.mean()))
                    add(li, "lambda_mean", float(lam.mean()))
                    add(li, "s_mean", float(s.mean()))
        return fn

    def block_hook(li):
        def fn(mod, inputs, output):
            x = output.float()
            ax = x.abs()
            mx = ax.max()
            med = ax.median()
            flat = x.flatten()
            mu, sd = flat.mean(), flat.std()
            kurt = float(((flat - mu) / sd).pow(4).mean())
            pos = torch.nonzero(ax == mx)[0].tolist()
            add(li, "max_act", float(mx))
            add(li, "median_act", float(med))
            add(li, "max_over_median", float(mx / med.clamp_min(1e-12)))
            add(li, "kurtosis", kurt)
            add(li, "argmax_pos", pos[1])
            add(li, "argmax_channel", pos[2])
            per_tok_max = ax.amax(-1)
            add(li, "max_act_pos0", float(per_tok_max[:, 0].max()))
            add(li, "max_act_pos_gt0", float(per_tok_max[:, 1:].max()))
        return fn

    for li, blk in enumerate(model.blocks):
        hooks.append(blk.attn.register_forward_hook(attn_hook(li)))
        hooks.append(blk.register_forward_hook(block_hook(li)))
    try:
        for d in batches:
            cur["special"] = d["special"]
            with autocast(dtype):
                model(d["input"], ctx=Ctx(record=True, fast=False, special=d["special"]))
    finally:
        for h in hooks:
            h.remove()
    layers = []
    for li in range(L):
        row = {}
        for k, v in acc[li].items():
            if k in ("argmax_pos", "argmax_channel"):
                row[k] = v
            elif k.endswith("_by_pos"):
                row[k] = {b: float(np.mean([x[b] for x in v])) for b in v[0]}
            elif k in ("max_act", "max_over_median", "max_act_pos0", "max_act_pos_gt0"):
                row[k] = float(np.max(v))
            else:
                row[k] = float(np.mean(v))
        layers.append(row)
    out = {"layers": layers, "n_windows": 4 * len(batches)}
    mx = [l["max_act"] for l in layers]
    out["max_act"] = float(max(mx))
    out["max_act_layer"] = int(np.argmax(mx)) + 1
    out["max_over_median"] = float(max(l["max_over_median"] for l in layers))
    sp = torch.cat([d["special"] for d in batches]).double().cpu()
    inv = 1.0 / torch.arange(1, sp.shape[1] + 1, dtype=torch.float64)
    pu = (sp.cumsum(-1) * inv).mean(0)
    out["uniform"] = {"first_mass": float(inv[1:].mean()), "punct_mass": float(pu[1:].mean()),
                      "first_mass_by_pos": _bucket_means(inv, POS_BUCKETS),
                      "punct_mass_by_pos": _bucket_means(pu, POS_BUCKETS)}
    if sp.shape[1] > FAR:
        out["uniform"].update({f"first_mass_ge{FAR}": float(inv[FAR:].mean()),
                               f"punct_mass_ge{FAR}": float(pu[FAR:].mean())})
    log(f"[stats] max act {out['max_act']:.1f} (layer {out['max_act_layer']}), max/median {out['max_over_median']:.1f}")

    att = model.blocks[0].attn
    if getattr(att, "cfg", None) is not None and (isinstance(att, SinkAttention) or
                                                   (isinstance(att, AbsorbAttention) and att.cfg.null_mode != "none")):
        longd = list(data.fixed_batches(1, a.stats_long_windows, scale=a.stats_long_scale, device=device))
        per_layer = [torch.zeros(a.train_len * a.stats_long_scale, device=device) for _ in range(L)]
        hk = []

        def null_hook(li):
            def fn(mod, inputs, output):
                n = output[1].get("null")
                if n is not None:
                    per_layer[li] += n.float().mean(dim=(0, 1))
            return fn

        for li, blk in enumerate(model.blocks):
            hk.append(blk.attn.register_forward_hook(null_hook(li)))
        try:
            for d in longd:
                with autocast(dtype):
                    hidden(model, d["input"], a.chunk)
        finally:
            for h in hk:
                h.remove()
        out["null_by_pos_long"] = [_bucket_means(v / len(longd), POS_BUCKETS) for v in per_layer]
    return out


def run_params(model, a, device, dtype, log):
    rows = []
    for li, blk in enumerate(model.blocks):
        att = blk.attn
        r = {}
        if hasattr(att, "sink"):
            r["sink_logit"] = att.sink.detach().float().cpu().tolist()
        if hasattr(att, "null_logit"):
            r["static_null_logit"] = att.null_logit.detach().float().cpu().tolist()
        if hasattr(att, "null_proj"):
            r["null_bias"] = att.null_proj.bias.detach().float().cpu().tolist()
            r["null_weight_norm"] = att.null_proj.weight.detach().float().norm(dim=1).cpu().tolist()
        if hasattr(att, "rho_proj"):
            r["rho_bias"] = att.rho_proj.bias.detach().float().cpu().tolist()
            r["rho_weight_norm"] = att.rho_proj.weight.detach().float().norm(dim=1).cpu().tolist()
        if hasattr(att, "q_norm"):
            g = (att.q_norm.weight * att.k_norm.weight).detach().float()
            r["qk_gain_mean"] = float(g.mean())
            r["qk_gain_rms"] = float(g.pow(2).mean().sqrt())
        if hasattr(blk, "delta"):
            r["beta_bias"] = float(blk.delta.proj.bias.detach().float())
            r["beta_weight_norm"] = float(blk.delta.proj.weight.detach().float().norm())
        if hasattr(blk, "gain"):
            r["gain_bias"] = float(blk.gain.proj.bias.detach().float())
            r["gain_weight_norm"] = float(blk.gain.proj.weight.detach().float().norm())
        rows.append(r)
    return {"layers": rows}


QUANT_LINEARS = ("attn.qkv", "attn.o", "mlp.up", "mlp.down")


def quant_targets(model, scope="all"):
    if scope not in ("all", "proj"):
        raise ValueError(scope)
    out = []
    for name, mod in model.named_modules():
        rel = name.split(".", 2)[-1]
        if isinstance(mod, nn.Linear) and name.startswith("blocks.") and (scope == "all" or rel in QUANT_LINEARS):
            out.append((rel, mod))
    return out


def _fq(x, dim, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def fq_act(x, act, bits=8, skip0=False):
    xf = x.float()
    if not skip0:
        dims = tuple(range(xf.dim())) if act == "tensor" else (-1,)
        return _fq(xf, dims, bits).to(x.dtype)
    if xf.dim() != 3:
        raise ValueError(f"skip0 needs a (batch, position, channel) input, got shape {tuple(xf.shape)}")
    out = xf.clone()
    if xf.shape[1] > 1:
        rest = xf[:, 1:]
        out[:, 1:] = _fq(rest, tuple(range(rest.dim())) if act == "tensor" else (-1,), bits)
    return out.to(x.dtype)


@contextlib.contextmanager
def fake_quant(model, act="tensor", w_bits=8, a_bits=8, scope="all", skip0=False):
    saved, hooks = [], []
    try:
        for _, mod in quant_targets(model, scope):
            saved.append((mod, mod.weight.data.clone()))
            mod.weight.data = _fq(mod.weight.data.float(), dim=1, bits=w_bits).to(mod.weight.dtype)
            if act is not None:
                def pre(m, inputs, act=act):
                    return (fq_act(inputs[0], act, a_bits, skip0),) + tuple(inputs[1:])
                hooks.append(mod.register_forward_pre_hook(pre))
        yield sorted({n for n, _ in quant_targets(model, scope)})
    finally:
        for h in hooks:
            h.remove()
        for mod, w in saved:
            mod.weight.data = w


def val_ppl(model, batches, chunk):
    tot, cnt = 0.0, 0
    for d in batches:
        tot += float(full_nll(model, d["input"], d["target"], None, chunk).sum())
        cnt += d["target"].numel()
    return math.exp(tot / cnt)


@torch.no_grad()
def run_quant(model, a, device, dtype, log):
    from tasks import LMData
    data = LMData(a.data_dir, seq_len=a.train_len)
    batches = list(data.fixed_batches(8, a.quant_batches, device=device))

    def ppl():
        return val_ppl(model, batches, a.chunk)

    base = ppl()
    out = {"fp32": base, "tokens": sum(d["target"].numel() for d in batches)}
    modes = (("w8", None), ("w8a8_token", "token"), ("w8a8_tensor", "tensor"))
    for scope, suf in (("all", ""), ("proj", "_proj")):
        mods = sorted({n for n, _ in quant_targets(model, scope)})
        out["quantised_modules" + suf] = mods
        if suf and mods == out["quantised_modules"]:
            out.update({k + suf + x: out[k + x] for k, _ in modes for x in ("", "_dppl_pct")})
            continue
        for name, act in modes:
            with fake_quant(model, act=act, scope=scope):
                out[name + suf] = ppl()
            out[name + suf + "_dppl_pct"] = 100 * (out[name + suf] / base - 1)
    log(f"[quant] fp32 {base:.3f} w8 {out['w8']:.3f} w8a8/token {out['w8a8_token']:.3f} "
        f"w8a8/tensor {out['w8a8_tensor']:.3f} (projections only: {out['w8a8_tensor_proj']:.3f}); "
        f"quantised {','.join(out['quantised_modules'])}")
    return out


def act_stats(x, skip0=False):
    x = x.float()
    if skip0:
        x = x[:, 1:]
    ax = x.abs()
    mx, med = ax.max(), ax.median()
    flat = x.flatten()
    mu, sd = flat.mean(), flat.std()
    return {"max_act": float(mx), "median_act": float(med), "max_over_median": float(mx / med.clamp_min(1e-12)),
            "kurtosis": float(((flat - mu) / sd).pow(4).mean())}


def argmax_positions(x):
    i = x.float().abs().flatten(1).argmax(-1)
    D = x.shape[-1]
    return (i // D).tolist(), (i % D).tolist()


@torch.no_grad()
def run_pos0(model, a, device, dtype, log):
    from tasks import LMData
    data = LMData(a.data_dir, seq_len=a.train_len)
    L = len(model.blocks)
    acc = [dict() for _ in range(L)]

    def block_hook(li):
        def fn(mod, inputs, output):
            for suf, skip in (("", False), ("_excl0", True)):
                for k, v in act_stats(output, skip).items():
                    acc[li].setdefault(k + suf, []).append(v)
            pos, ch = argmax_positions(output)
            acc[li].setdefault("argmax_pos", []).extend(pos)
            acc[li].setdefault("argmax_channel", []).extend(ch)
        return fn

    batches = list(data.fixed_batches(4, a.stats_batches, device=device))
    hooks = [blk.register_forward_hook(block_hook(li)) for li, blk in enumerate(model.blocks)]
    try:
        for d in batches:
            with autocast(dtype):
                model(d["input"], ctx=Ctx(record=True, fast=False, special=d["special"]))
    finally:
        for h in hooks:
            h.remove()
    layers = []
    for li in range(L):
        row = {}
        for k, v in acc[li].items():
            if k in ("argmax_pos", "argmax_channel"):
                row[k] = v
            elif k.startswith(("max_act", "max_over_median")):
                row[k] = float(np.max(v))
            else:
                row[k] = float(np.mean(v))
        row["argmax_pos0_share"] = float(np.mean([p == 0 for p in row["argmax_pos"]]))
        layers.append(row)
    out = {"layers": layers, "n_windows": 4 * len(batches)}
    for suf in ("", "_excl0"):
        mx = [r["max_act" + suf] for r in layers]
        out["max_act" + suf] = float(max(mx))
        out["max_act_layer" + suf] = int(np.argmax(mx)) + 1
        out["max_over_median" + suf] = float(max(r["max_over_median" + suf] for r in layers))
        out["kurtosis_max" + suf] = float(max(r["kurtosis" + suf] for r in layers))
    out["argmax_pos0_share"] = float(np.mean([r["argmax_pos0_share"] for r in layers]))

    qb = list(data.fixed_batches(8, a.quant_batches, device=device))
    base = val_ppl(model, qb, a.chunk)
    q = {"fp32": base, "tokens": sum(d["target"].numel() for d in qb), "n_windows": 8 * len(qb)}
    modes = (("w8a8_token", "token"), ("w8a8_tensor", "tensor"))
    for scope, suf in (("all", ""), ("proj", "_proj")):
        mods = sorted({n for n, _ in quant_targets(model, scope)})
        q["quantised_modules" + suf] = mods
        if suf and mods == q["quantised_modules"]:
            q.update({k + suf + x: q[k + x] for k, _ in modes for x in ("", "_excl0", "_dppl_pct", "_excl0_dppl_pct")})
            continue
        for name, act in modes:
            for x, skip0 in (("", False), ("_excl0", True)):
                with fake_quant(model, act=act, scope=scope, skip0=skip0):
                    q[name + suf + x] = val_ppl(model, qb, a.chunk)
                q[name + suf + x + "_dppl_pct"] = 100 * (q[name + suf + x] / base - 1)
    out["quant"] = q
    log(f"[pos0] max act {out['max_act']:.1f} -> {out['max_act_excl0']:.1f} without position 0, max kurtosis "
        f"{out['kurtosis_max']:.1f} -> {out['kurtosis_max_excl0']:.1f}, maximum at position 0 in "
        f"{100 * out['argmax_pos0_share']:.0f}% of (layer, window); w8a8/tensor {q['w8a8_tensor_dppl_pct']:+.2f}% -> "
        f"{q['w8a8_tensor_excl0_dppl_pct']:+.2f}%, w8a8/token {q['w8a8_token_dppl_pct']:+.2f}% -> "
        f"{q['w8a8_token_excl0_dppl_pct']:+.2f}%")
    return out


GATE_BUCKETS = [(0, 1), (1, 16), (16, 256), (256, 1024)]
SCALE_KEYS = ("mean", "lt0.1", "lt0.5", "gt0.9")
GATE_KEYS = ("mean", "lt0.1", "gt0.9")
VAR_KEYS = ("head", "channel", "token", "token_head", "token_channel")


def gate_mechanism(model):
    from attention import AbsorbAttention, GatedAttention, SinkAttention, StickBreakingAttention
    att = model.blocks[0].attn
    null = gate = None
    if isinstance(att, AbsorbAttention):
        if att.cfg.read_gate:
            return None
        null = None if att.cfg.null_mode == "none" else att.cfg.null_mode
        gate = None if att.cfg.out_gate == "none" else att.cfg.out_gate
    elif isinstance(att, GatedAttention):
        gate = "head" if att.headwise else "elem"
    elif isinstance(att, SinkAttention):
        null = "sink"
    elif isinstance(att, StickBreakingAttention):
        null = "stickbreak"
    return {"null": null, "gate": gate} if (null or gate) else None


def gate_linear_of(att):
    from attention import GatedAttention
    return att.gate if isinstance(att, GatedAttention) else att.out_gate


def gate_values_from_logit(z, n_head):
    B, N, C = z.shape
    return torch.sigmoid(z).view(B, N, n_head, C // n_head).transpose(1, 2)


def _scale_block(acc, n, per_head=False):
    H, N = acc.shape[1:]
    out = dict(zip(SCALE_KEYS, (acc.sum(dim=(1, 2)) / (n * H * N)).tolist()))
    out["by_pos"] = {f"{lo}-{hi}": dict(zip(SCALE_KEYS, (acc[:, :, lo:min(hi, N)].sum(dim=(1, 2))
                                                         / (n * H * (min(hi, N) - lo))).tolist()))
                     for lo, hi in GATE_BUCKETS if lo < N}
    if per_head:
        out["per_head"] = dict(zip(SCALE_KEYS, (acc.sum(-1) / (n * N)).tolist()))
    return out


def logit_variance(z1, z2, zh, tokens):
    H, C = z1.shape
    m_hc = z1 / tokens
    m_h = m_hc.mean(-1)
    mu2 = float(m_h.mean()) ** 2
    e2 = float(z2.sum()) / (tokens * H * C)
    e_h, e_hc = float(m_h.pow(2).mean()), float(m_hc.pow(2).mean())
    total = e2 - mu2
    if not total > 1e-9 * max(e2, 1e-300):
        return {"total": max(total, 0.0), **dict.fromkeys(VAR_KEYS)}
    tok, tok_head = e2 - e_hc, float(zh.sum()) / (tokens * H) - e_h
    parts = (e_h - mu2, e_hc - e_h, tok, tok_head, tok - tok_head)
    return {"total": total, **{k: max(v, 0.0) / total for k, v in zip(VAR_KEYS, parts)}}


def _worst(cur, err):
    return err if cur is None or err != err or err > cur else cur


class GateProbe:
    def __init__(self, model, mech, keep=False):
        self.model, self.mech = model, mech
        att = model.blocks[0].attn
        self.L, self.H, self.dh = len(model.blocks), att.n_head, att.d_head
        self.both = bool(mech["null"] and mech["gate"])
        self.scale = [dict() for _ in range(self.L)]
        self.gate = [dict() for _ in range(self.L)]
        self.n_windows, self.N = 0, None
        self.check = {"gate_recon_rel_err_max": 0.0 if mech["gate"] else None, "null_sum_abs_err_max": None}
        self._cur, self._hooks = {}, []
        self.keep = keep
        self.kept = [dict() for _ in range(self.L)]

    def __enter__(self):
        for li, blk in enumerate(self.model.blocks):
            att = blk.attn
            if self.mech["gate"]:
                self._hooks.append(att.qkv.register_forward_hook(self._keep(li, "qkv")))
                self._hooks.append(gate_linear_of(att).register_forward_hook(self._keep(li, "z")))
                self._hooks.append(att.o.register_forward_pre_hook(self._keep_input(li, "merged")))
            self._hooks.append(att.register_forward_hook(self._attention(li)))
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks, self._cur = [], {}
        return False

    def _keep(self, li, name):
        def fn(mod, inputs, output):
            self._cur[(li, name)] = output
        return fn

    def _keep_input(self, li, name):
        def fn(mod, inputs):
            self._cur[(li, name)] = inputs[0]
        return fn

    def _attention(self, li):
        def fn(mod, inputs, output):
            aux = output[1]
            a_ = aux.get("attn")
            if a_ is None:
                raise RuntimeError("the gate statistics need a recorded forward pass (Ctx(record=True))")
            B, H, N = a_.shape[:3]
            if self.N is None:
                self.N = N
            elif N != self.N:
                raise RuntimeError(f"windows of {N} tokens after windows of {self.N}")
            if li == 0:
                self.n_windows += B
            keys = a_.sum(-1).double()
            if aux.get("null") is not None:
                err = float((keys + aux["null"].double() - 1.0).abs().max())
                self.check["null_sum_abs_err_max"] = _worst(self.check["null_sum_abs_err_max"], err)
            fac = {}
            if self.mech["null"]:
                fac["scale_null"] = keys
            if self.mech["gate"]:
                qkv, z, merged = (self._cur.pop((li, k)) for k in ("qkv", "z", "merged"))
                D = qkv.shape[-1] // 3
                v = qkv[..., 2 * D:].view(B, N, H, D // H).transpose(1, 2)
                o = mod.mix(a_, v)
                g = gate_values_from_logit(z, H)
                y = o * g.to(o.dtype)
                ref = merged.double()
                err = float((y.transpose(1, 2).reshape(B, N, D).double() - ref).norm()
                            / ref.norm().clamp_min(1e-300))
                self.check["gate_recon_rel_err_max"] = _worst(self.check["gate_recon_rel_err_max"], err)
                fac["scale_gate"] = y.double().norm(dim=-1) / o.double().norm(dim=-1).clamp_min(1e-300)
                self._add_gate(li, g.double(), z.double().reshape(B * N, H, -1))
            vals = dict(fac) if self.both else {}
            vals["scale"] = fac["scale_null"] * fac["scale_gate"] if self.both else next(iter(fac.values()))
            for k, x in vals.items():
                s = torch.stack([x, (x < 0.1).double(), (x < 0.5).double(), (x > 0.9).double()]).sum(1)
                self.scale[li][k] = self.scale[li][k] + s if k in self.scale[li] else s
            if self.keep:
                for k, x in vals.items():
                    self.kept[li].setdefault(k, []).append(x.detach().to("cpu", torch.float64))
        return fn

    def _add_gate(self, li, g, z):
        new = {"g": torch.stack([g.sum(dim=(0, 3)), (g < 0.1).double().sum(dim=(0, 3)),
                                 (g > 0.9).double().sum(dim=(0, 3))]),
               "z1": z.sum(0), "z2": z.pow(2).sum(0), "zh": z.mean(-1).pow(2).sum(0), "tokens": z.shape[0]}
        acc = self.gate[li]
        for k, x in new.items():
            acc[k] = acc[k] + x if k in acc else x

    def _gate_block(self, acc):
        W, N, H = self.n_windows, self.N, self.H
        C = acc["z1"].shape[1]
        g = acc["g"]
        out = dict(zip(GATE_KEYS, (g.sum(dim=(1, 2)) / (W * N * H * C)).tolist()))
        out["per_head_mean"] = (g[0].sum(-1) / (W * N * C)).tolist()
        out["by_pos"] = {f"{lo}-{hi}": float(g[0][:, lo:min(hi, N)].sum() / (W * H * C * (min(hi, N) - lo)))
                         for lo, hi in GATE_BUCKETS if lo < N}
        out["logit_mean"] = float(acc["z1"].sum() / (acc["tokens"] * H * C))
        out["logit_var"] = logit_variance(acc["z1"], acc["z2"], acc["zh"], acc["tokens"])
        return out

    def result(self):
        if not self.n_windows:
            raise RuntimeError("no forward pass was recorded")
        W, N, L, H = self.n_windows, self.N, self.L, self.H
        kinds = ["scale"] + (["scale_null", "scale_gate"] if self.both else [])
        out = {"mechanism": dict(self.mech), "n_windows": W, "seq_len": N, "n_layer": L, "n_head": H,
               "d_head": self.dh, "pos_buckets": [f"{lo}-{hi}" for lo, hi in GATE_BUCKETS if lo < N]}
        layers = [dict() for _ in range(L)]
        for k in kinds:
            accs = [self.scale[li][k] for li in range(L)]
            out[k] = _scale_block(torch.stack(accs).sum(0), W * L)
            for li in range(L):
                layers[li][k] = _scale_block(accs[li], W, per_head=True)
            ph = torch.stack([x[0].sum(-1) / (W * N) for x in accs])
            out["heads" + k[len("scale"):]] = {"n": L * H, "lt0.05": int((ph < 0.05).sum()),
                                               "lt0.5": int((ph < 0.5).sum()), "gt0.95": int((ph > 0.95).sum())}
        if self.mech["gate"]:
            for li in range(L):
                layers[li]["gate"] = self._gate_block(self.gate[li])
            rows = [r["gate"] for r in layers]
            pooled = {k: float(np.mean([r[k] for r in rows])) for k in GATE_KEYS}
            pooled["logit_var"] = {}
            for k in ("total",) + VAR_KEYS:
                vals = [r["logit_var"][k] for r in rows if r["logit_var"][k] is not None]
                pooled["logit_var"][k] = float(np.mean(vals)) if vals else None
            out["gate"] = pooled
        out["check"] = dict(self.check)
        out["layers"] = layers
        return out


@torch.no_grad()
def run_gate(model, a, device, dtype, log):
    mech = gate_mechanism(model)
    if mech is None:
        log("[gate] no output gate and no null slot (a read gate is not covered): nothing to measure")
        return {}
    from tasks import LMData
    data = LMData(a.data_dir, seq_len=a.train_len)
    batches = list(data.fixed_batches(4, a.stats_batches, device=device))
    with GateProbe(model, mech) as probe:
        for d in batches:
            with autocast(dtype):
                model(d["input"], ctx=Ctx(record=True, fast=False, special=d["special"]))
    out = probe.result()
    sc, hd = out["scale"], out["heads"]
    msg = (f"[gate] null={mech['null']} gate={mech['gate']}: output scale {sc['mean']:.3f} (<0.1: "
           f"{100 * sc['lt0.1']:.1f}%, <0.5: {100 * sc['lt0.5']:.1f}%, >0.9: {100 * sc['gt0.9']:.1f}%); heads with a "
           f"mean scale <0.05: {hd['lt0.05']}, <0.5: {hd['lt0.5']}, >0.95: {hd['gt0.95']} of {hd['n']}")
    if "scale_null" in out:
        msg += f"; null factor {out['scale_null']['mean']:.3f}, gate factor {out['scale_gate']['mean']:.3f}"
    if "gate" in out:
        gv, lv = out["gate"], out["gate"]["logit_var"]
        msg += f"; gate values {gv['mean']:.3f} (<0.1: {100 * gv['lt0.1']:.1f}%, >0.9: {100 * gv['gt0.9']:.1f}%)"
        if lv["head"] is not None:
            msg += (f", logit variance: head {lv['head']:.2f}, channel {lv['channel']:.2f}, token {lv['token']:.2f} "
                    f"(shared by a head's channels: {lv['token_head']:.2f})")
        msg += f"; recomputation error {out['check']['gate_recon_rel_err_max']:.1e}"
    log(msg)
    return out


REL_THRESHOLDS = (0.5, 0.25)
REL_KEYS = tuple(f"rel_lt{t:g}" for t in REL_THRESHOLDS)
SCALE_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
REL_HIST_LOG2 = tuple(k / 2 for k in range(-20, 9))
JOINT_KEYS = ("both", "null_only", "gate_only")


def relative_scale(x, med):
    m = med.view(1, -1, 1)
    pos = m > 0
    return torch.where(pos, x / torch.where(pos, m, torch.ones_like(m)),
                       torch.where(x > 0, torch.full_like(x, math.inf), torch.ones_like(x)))


def rel_hist(r, H):
    bounds = torch.tensor([2.0 ** e for e in REL_HIST_LOG2], dtype=torch.float64)
    idx = torch.bucketize(r.transpose(0, 1).reshape(H, -1).contiguous(), bounds, right=True)
    out = torch.zeros(H, len(bounds) + 1, dtype=torch.int64)
    return out.scatter_add_(1, idx, torch.ones_like(idx))


def _spread(vals):
    v = [x for x in vals if x is not None]
    return {"mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}


def _bucket_counts(lt, N):
    return {f"{lo}-{hi}": lt[:, :, lo:min(hi, N)].sum(dim=(0, 2)) for lo, hi in GATE_BUCKETS if lo < N}


def gate2_layer(x):
    W, H, N = x.shape
    vals = x.transpose(0, 1).reshape(H, -1).numpy()
    qs = np.quantile(vals, SCALE_QUANTILES, axis=1)
    q = dict(zip(SCALE_QUANTILES, qs))
    med = torch.from_numpy(np.ascontiguousarray(q[0.5]))
    r = relative_scale(x, med)
    sd_log = []
    for v in vals:
        pos = v[v > 0]
        sd_log.append(float(np.log(pos).std()) if pos.size else None)
    iqr = [float((q[0.75][h] - q[0.25][h]) / q[0.5][h]) if q[0.5][h] > 0 else None for h in range(H)]
    n_head = W * N
    width = {f"{lo}-{hi}": min(hi, N) - lo for lo, hi in GATE_BUCKETS if lo < N}
    hist = rel_hist(r, H)
    cnt = {"n": W * N * H, "by_pos_n": {b: W * w * H for b, w in width.items()}, "hist": hist.sum(0),
           "lt": {}, "by_pos": {b: {} for b in width}, "iqr": iqr, "sd_log": sd_log, "median": q[0.5].tolist(),
           "zero": int((x == 0).sum()), "median_zero": int((med == 0).sum())}
    per_head = {"median": q[0.5].tolist(), "mean": vals.mean(axis=1).tolist(),
                "quantiles": {f"{p:g}": q[p].tolist() for p in SCALE_QUANTILES}, "iqr_over_median": iqr,
                "sd_log": sd_log, "n_zero": (x == 0).sum(dim=(0, 2)).tolist(), "by_pos": {b: {} for b in width},
                "hist": hist.tolist()}
    for t, k in zip(REL_THRESHOLDS, REL_KEYS):
        lt = r < t
        per_head[k] = (lt.sum(dim=(0, 2)).double() / n_head).tolist()
        cnt["lt"][k] = int(lt.sum())
        for b, c in _bucket_counts(lt, N).items():
            per_head["by_pos"][b][k] = (c.double() / (W * width[b])).tolist()
            cnt["by_pos"][b][k] = int(c.sum())
    return _gate2_block(cnt), dict(cnt, per_head=per_head), r


def _gate2_block(cnt):
    out = {k: cnt["lt"][k] / cnt["n"] for k in REL_KEYS}
    out["by_pos"] = {b: {k: v[k] / cnt["by_pos_n"][b] for k in REL_KEYS} for b, v in cnt["by_pos"].items()}
    out["hist"] = [int(c) for c in cnt["hist"]]
    out["dispersion"] = {"iqr_over_median": _spread(cnt["iqr"]), "sd_log": _spread(cnt["sd_log"])}
    out["head_median"] = _spread(cnt["median"])
    return out


def _pool_counts(cnts):
    first = cnts[0]
    return {"n": sum(c["n"] for c in cnts), "by_pos_n": {b: sum(c["by_pos_n"][b] for c in cnts) for b in first["by_pos_n"]},
            "hist": sum(c["hist"] for c in cnts), "lt": {k: sum(c["lt"][k] for c in cnts) for k in REL_KEYS},
            "by_pos": {b: {k: sum(c["by_pos"][b][k] for c in cnts) for k in REL_KEYS} for b in first["by_pos"]},
            "iqr": [v for c in cnts for v in c["iqr"]], "sd_log": [v for c in cnts for v in c["sd_log"]],
            "median": [v for c in cnts for v in c["median"]], "zero": sum(c["zero"] for c in cnts),
            "median_zero": sum(c["median_zero"] for c in cnts)}


def joint_counts(r_null, r_gate):
    out = {}
    for t, k in zip(REL_THRESHOLDS, REL_KEYS):
        a, b = r_null < t, r_gate < t
        out[k] = {"both": int((a & b).sum()), "null_only": int((a & ~b).sum()), "gate_only": int((~a & b).sum())}
    return out


def gate2_result(probe):
    if not probe.n_windows:
        raise RuntimeError("no forward pass was recorded")
    W, N, L, H = probe.n_windows, probe.N, probe.L, probe.H
    kinds = ["scale"] + (["scale_null", "scale_gate"] if probe.both else [])
    out = {"mechanism": dict(probe.mech), "n_windows": W, "seq_len": N, "n_layer": L, "n_head": H,
           "d_head": probe.dh, "pos_buckets": [f"{lo}-{hi}" for lo, hi in GATE_BUCKETS if lo < N], "kinds": kinds,
           "thresholds": list(REL_THRESHOLDS), "quantiles": list(SCALE_QUANTILES),
           "hist_log2_edges": list(REL_HIST_LOG2)}
    layers = [dict() for _ in range(L)]
    zero, median_zero = {}, {}
    rel = [dict() for _ in range(L)]
    for k in kinds:
        cnts = []
        for li in range(L):
            x = torch.cat(probe.kept[li][k])
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError(f"non-finite output scales ({k}, layer {li})")
            block, cnt, r = gate2_layer(x)
            block["per_head"] = cnt["per_head"]
            layers[li][k] = block
            cnts.append(cnt)
            if k != "scale":
                rel[li][k] = r
        pooled = _pool_counts(cnts)
        out[k] = _gate2_block(pooled)
        zero[k], median_zero[k] = pooled["zero"], pooled["median_zero"]
    if probe.both:
        total = {k: {j: 0 for j in JOINT_KEYS} for k in REL_KEYS}
        for li in range(L):
            jc = joint_counts(rel[li]["scale_null"], rel[li]["scale_gate"])
            layers[li]["joint"] = {k: {j: v / (W * N * H) for j, v in jc[k].items()} for k in REL_KEYS}
            for k in REL_KEYS:
                for j in JOINT_KEYS:
                    total[k][j] += jc[k][j]
        out["joint"] = {k: {j: v / (W * N * H * L) for j, v in total[k].items()} for k in REL_KEYS}
    out["check"] = dict(probe.check, zero_scale=zero, median_zero_heads=median_zero)
    out["layers"] = layers
    return out


@torch.no_grad()
def run_gate2(model, a, device, dtype, log):
    mech = gate_mechanism(model)
    if mech is None:
        log("[gate2] no output gate and no null slot (a read gate is not covered): nothing to measure")
        return {}
    from tasks import LMData
    data = LMData(a.data_dir, seq_len=a.train_len)
    batches = list(data.fixed_batches(4, a.stats_batches, device=device))
    with GateProbe(model, mech, keep=True) as probe:
        for d in batches:
            with autocast(dtype):
                model(d["input"], ctx=Ctx(record=True, fast=False, special=d["special"]))
    out = gate2_result(probe)
    msg = f"[gate2] null={mech['null']} gate={mech['gate']}:"
    for k in out["kinds"]:
        b = out[k]
        sd = b["dispersion"]["sd_log"]["median"]
        msg += (f" {k}: relative scale < 1/2 {100 * b['rel_lt0.5']:.1f}%, < 1/4 {100 * b['rel_lt0.25']:.1f}% "
                f"(position 0: {100 * b['by_pos']['0-1']['rel_lt0.5']:.1f}%), per-head s.d. of log scale median "
                f"{'n/a' if sd is None else f'{sd:.3f}'};")
    if "joint" in out:
        j = out["joint"]["rel_lt0.5"]
        msg += (f" both factors < 1/2 {100 * j['both']:.1f}%, null only {100 * j['null_only']:.1f}%, gate only "
                f"{100 * j['gate_only']:.1f}%;")
    log(msg.rstrip(";"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prefetch", action="store_true", help="only build the long-document cache, then exit")
    ap.add_argument("--what", default="params,stats,window,quant,longdoc,rope,evict",
                    help="comma list of parts: params, stats, window, quant, longdoc, rope, pos0, gate, gate2, "
                         "evict, evict_matched (after evict)")
    ap.add_argument("--data_dir", default="data/fineweb_edu")
    ap.add_argument("--cache_dir", default="data/longdocs")
    ap.add_argument("--scales", default="2,4,8,16")
    ap.add_argument("--budget", type=int, default=131072, help="validation tokens per length (as long_context.py)")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--n_sink", type=int, default=4)
    ap.add_argument("--longdoc_n", type=int, default=32)
    ap.add_argument("--longdoc_len", type=int, default=16384)
    ap.add_argument("--longdoc_sources", default="emozilla/pg19,deepmind/pg19,fineweb",
                    help="Hub datasets tried in order (test split); 'fineweb' = long validation documents")
    ap.add_argument("--evict_scales", default="4:2x2,16:2x1", help="scale:BATCHxN_BATCHES per length")
    ap.add_argument("--stats_batches", type=int, default=4, help="batches of 4 windows of 1k")
    ap.add_argument("--stats_long_scale", type=int, default=16)
    ap.add_argument("--stats_long_windows", type=int, default=2)
    ap.add_argument("--quant_batches", type=int, default=4, help="batches of 8 windows of 1k")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from zeroshot import load_model
    if a.prefetch:
        ok = False
        try:
            toks, source = load_longdocs(a.data_dir, a.cache_dir, a.longdoc_n, a.longdoc_len, print,
                                         tuple(x for x in a.longdoc_sources.split(",") if x and x != "fineweb"))
            print(f"long documents: {toks.shape} from {source}")
            ok = True
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if ok else 1)
    if not a.ckpt:
        ap.error("--ckpt is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model, ck = load_model(a.ckpt, device)
    a.train_len = ck["args"].get("seq_len", 1024)
    out_path = a.out or os.path.join(os.path.dirname(a.ckpt), "internals_eval.json")
    what = a.what.split(",")
    if "evict_matched" in what:
        what = [w for w in what if w != "evict_matched"] + ["evict_matched"]
    t0 = time.time()

    def log(msg):
        print(msg, flush=True)

    res = load_json(out_path, log)
    res.update({"ckpt": a.ckpt, "variant": ck["args"].get("variant"), "seed": ck["args"].get("seed"),
                "train_len": a.train_len, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})

    def save():
        save_json(res, out_path)

    fns = {"params": run_params, "stats": run_stats, "window": run_window, "quant": run_quant,
           "longdoc": run_longdoc, "rope": run_rope, "pos0": run_pos0, "gate": run_gate, "gate2": run_gate2,
           "evict": lambda *x: run_evict(*x, res=res, save=save),
           "evict_matched": lambda *x: run_evict_matched(*x, res=res, save=save)}
    failed = []
    for w in what:
        if w in res:
            continue
        t = time.time()
        try:
            res[w] = fns[w](model, a, device, dtype, log)
        except Exception:
            traceback.print_exc()
            failed.append(w)
        if w in failed:
            model.eval()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if w in ("evict", "evict_matched"):
            res.pop(w + "_partial", None)
            res.pop(w + "_running", None)
        res.setdefault("seconds", {})[w] = time.time() - t
        save()
    res["done"] = all(k in res for k in what)
    save()
    log(f"wrote {out_path} ({time.time() - t0:.0f}s)" + (f"; FAILED parts: {','.join(failed)}" if failed else ""))
    if _STREAMED:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failed else 0)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
