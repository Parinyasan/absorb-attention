import math
from contextlib import nullcontext

import torch

from model import Ctx

EVAL_SEED = 1234


def autocast(device, dtype):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def eval_task(model, task, scales, n_batches, batch, device, amp_dtype):
    model.eval()
    out = {}
    for s in scales:
        gen = torch.Generator(device=device).manual_seed(EVAL_SEED + s)
        b = max(1, batch // s)
        correct = total = seq_ok = seqs = 0
        loss_sum = 0.0
        try:
            for _ in range(n_batches * max(1, batch // b)):
                d = task.sample(b, scale=s, gen=gen, device=device)
                with autocast(device, amp_dtype):
                    logits, loss = model(d["input"], d["target"])
                m = d["target"] != -100
                pred = logits.argmax(-1)
                hit = (pred == d["target"]) & m
                correct += int(hit.sum())
                total += int(m.sum())
                seq_ok += int(((hit | ~m).all(-1)).sum())
                seqs += b
                loss_sum += float(loss) * int(m.sum())
            out[f"acc@{s}x"] = correct / max(total, 1)
            out[f"seq_acc@{s}x"] = seq_ok / max(seqs, 1)
            out[f"loss@{s}x"] = loss_sum / max(total, 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out[f"acc@{s}x"] = None
    model.train()
    return out


@torch.no_grad()
def eval_lm(model, data, scales, n_batches, batch, device, amp_dtype):
    model.eval()
    out = {}
    for s in scales:
        b = max(1, batch // s)
        tot, cnt = 0.0, 0
        try:
            for d in data.fixed_batches(b, n_batches * max(1, batch // b), scale=s, device=device):
                with autocast(device, amp_dtype):
                    _, loss = model(d["input"], d["target"])
                n = d["target"].numel()
                tot += float(loss) * n
                cnt += n
            out[f"loss@{s}x"] = tot / max(cnt, 1)
            out[f"ppl@{s}x"] = math.exp(min(out[f"loss@{s}x"], 50))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out[f"loss@{s}x"] = None
    model.train()
    return out


@torch.no_grad()
def eval_stats(model, batch_dict, device, amp_dtype):
    model.eval()
    ctx = Ctx(record=True, fast=False, special=batch_dict["special"])
    with autocast(device, amp_dtype):
        model(batch_dict["input"], ctx=ctx)
    model.train()
    layers = ctx.stats
    summary = {}
    for key in ("null_mean", "beta_mean", "gate_mean", "first_mass", "punct_mass", "punct_uniform",
                "attn_entropy", "R_mean", "retired_frac_R>2", "R_bos_mean", "retired_frac_R_bos>2", "rho_mean",
                "max_act"):
        vals = [l[key] for l in layers if key in l]
        if vals:
            summary[key] = max(vals) if key == "max_act" else sum(vals) / len(vals)
    return {"summary": summary, "layers": layers}


def benchmark(model, vocab_size, seq_len, batch, device, amp_dtype, steps=10):
    if device.type != "cuda":
        return {}
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        with autocast(device, amp_dtype):
            _, loss = model(x, x)
        loss.backward()
        opt.zero_grad(set_to_none=True)
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / steps
    train_mem = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), autocast(device, amp_dtype):
        model(x[:1])
    infer_mem = torch.cuda.max_memory_allocated() / 2**20
    return {"train_ms_per_step": ms, "train_tok_per_s": batch * seq_len / (ms / 1000),
            "train_peak_MiB": train_mem, "infer_peak_MiB_b1": infer_mem}
