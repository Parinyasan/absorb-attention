import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from model import Ctx
from internals_eval import load_json, save_json
from zeroshot import load_model, score

FILLER = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
INTRO = ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. "
         "I will quiz you about the important information there.")
KEY = " The pass key is {k}. Remember it. {k} is the pass key."
ASK = " What is the pass key? The pass key is"


@torch.no_grad()
def eval_ppl(model, data_dir, train_len, scales, budget, chunk, device, dtype):
    from tasks import LMData
    out = {}
    for s in scales:
        n_tok = train_len * s
        data = LMData(data_dir, seq_len=train_len)
        n_win = max(1, budget // n_tok)
        b = max(1, min(8, n_win, 8192 // n_tok if n_tok <= 8192 else 1))
        tot, cnt = 0.0, 0
        for d in data.fixed_batches(b, math.ceil(n_win / b), scale=s, device=device):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits, _ = model(d["input"], ctx=Ctx(chunk=chunk))
            tot += float(F.cross_entropy(logits.float().flatten(0, 1), d["target"].flatten(), reduction="sum"))
            cnt += d["target"].numel()
        out[f"ppl@{s}x"] = math.exp(tot / cnt)
        out[f"tokens@{s}x"] = cnt
        print(f"[ppl] {s}x ({n_tok} tokens/window): {out[f'ppl@{s}x']:.3f}", flush=True)
    return out


def passkey_pairs(tok, length, depth, trials, seed):
    rng = random.Random(seed)
    intro, ask, fill = tok.encode(INTRO), tok.encode(ASK), tok.encode(FILLER)
    pairs = []
    for _ in range(trials):
        k = rng.randint(10000, 99999)
        key = tok.encode(KEY.format(k=k))
        n_fill = max(0, length - len(intro) - len(key) - len(ask) - 8)
        filler = (fill * (n_fill // len(fill) + 1))[:n_fill]
        cut = int(depth * len(filler))
        ctx = intro + filler[:cut] + key + filler[cut:] + ask
        pairs.append((ctx, tok.encode(f" {k}")))
    return pairs


@torch.no_grad()
def eval_passkey(model, tok, lengths, depths, trials, chunk, device, dtype):
    out = {}
    for L in lengths:
        hits, n = 0, 0
        per_depth = {}
        for d in depths:
            pairs = passkey_pairs(tok, L, d, trials, seed=1000 * L + int(100 * d))
            res = score(model, pairs, tok.eos_token_id, max_len=L + 64, batch=1 if L > 4096 else 4,
                        device=device, dtype=dtype, chunk=chunk)
            ok = sum(g for _, g in res)
            per_depth[str(d)] = ok / len(res)
            hits += ok
            n += len(res)
        out[f"passkey@{L}"] = hits / n
        out[f"passkey_depth@{L}"] = per_depth
        print(f"[passkey] {L} tokens: {hits}/{n}", flush=True)
    return out


def eval_evict(model, data_dir, train_len, device, dtype):
    from attention import AbsorbAttention, SelectiveAttention
    from inference import eval_eviction
    from tasks import LMData
    data = LMData(data_dir, seq_len=train_len)
    N = train_len
    settings = [{"policy": "none"}]
    for frac in (0.5, 0.25):
        b = int(N * frac)
        settings += [{"policy": "tova", "budget": b}, {"policy": "window", "budget": b, "n_sink": 4},
                     {"policy": "window", "budget": b, "n_sink": 0},
                     {"policy": "stream", "budget": b, "n_sink": 4}]
    att = model.blocks[0].attn
    if isinstance(att, SelectiveAttention) or (isinstance(att, AbsorbAttention) and model.cfg.retire):
        settings += [{"policy": "retire_budget", "budget": int(N * f)} for f in (0.5, 0.25)]
        settings += [{"policy": "retire_tau", "tau": t} for t in (1.0, 2.0, 4.0, 8.0)]
    res = eval_eviction(model, data, settings, batch=4, n_batches=2, device=str(device), amp_dtype=dtype)
    for r in res:
        print(f"[evict] {json.dumps(r)}", flush=True)
    return res


def evict_current(res):
    rows = (res.get("evict") or []) + (res.get("evict_nosink") or [])
    return bool(rows) and any(r.get("policy") == "stream" for r in rows) and all("evicted_mass_mean" in r for r in rows)


def split_evict(rows):
    free = [r["policy"] == "window" and r.get("n_sink", 4) == 0 for r in rows]
    return {"evict": [r for r, f in zip(rows, free) if not f], "evict_nosink": [r for r, f in zip(rows, free) if f]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--what", default="ppl,passkey,evict")
    ap.add_argument("--data_dir", default="data/fineweb_edu")
    ap.add_argument("--scales", default="1,2,4,8,16")
    ap.add_argument("--budget", type=int, default=131072, help="validation tokens per length")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--trials", type=int, default=10, help="prompts per (length, depth)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import GPT2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model, ck = load_model(a.ckpt, device)
    train_len = ck["args"].get("seq_len", 1024)
    out_path = a.out or os.path.join(os.path.dirname(a.ckpt), "long_context.json")
    res = load_json(out_path)
    res.update({"ckpt": a.ckpt, "variant": ck["args"].get("variant"), "seed": ck["args"].get("seed"),
                "train_len": train_len, "chunk": a.chunk,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    what = a.what.split(",")
    t0 = time.time()

    def save():
        save_json(res, out_path, indent=2)

    if "ppl" in what and "ppl" not in res:
        res["ppl"] = eval_ppl(model, a.data_dir, train_len, [int(x) for x in a.scales.split(",")], a.budget,
                              a.chunk, device, dtype)
        save()
    if "passkey" in what and "passkey" not in res:
        tok = GPT2TokenizerFast.from_pretrained("gpt2", token=os.environ.get("HF_TOKEN") or None)
        res["passkey"] = eval_passkey(model, tok, [int(x) for x in a.lengths.split(",")],
                                      [float(x) for x in a.depths.split(",")], a.trials, a.chunk, device, dtype)
        save()
    if "evict" in what and not evict_current(res):
        old = {k: res.pop(k) for k in ("evict", "evict_nosink") if k in res}
        if old:
            res.setdefault("evict_prev", old)
        res.update(split_evict(eval_evict(model, a.data_dir, train_len, device, dtype)))
        save()
    res["done"] = all(k in res for k in what) and ("evict" not in what or evict_current(res))
    save()
    print(f"wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
