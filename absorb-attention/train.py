import argparse
import json
import math
import os
import time

import torch

from eval import autocast, benchmark, eval_lm, eval_stats, eval_task
from model import Model, ModelConfig
from presets import PRESETS
from tasks import LMData, Vocab, make_task

MODEL_PRESETS = {
    "tiny": dict(n_layer=2, d_model=128, n_head=4),
    "lm50m": dict(n_layer=8, d_model=512, n_head=8),
    "lm100m": dict(n_layer=12, d_model=768, n_head=12),
}


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        else:
            v = {"true": True, "false": False}.get(v.lower(), v)
        out[k] = v
    return out


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mqar", choices=["mqar", "vt", "niah", "lm"])
    ap.add_argument("--variant", default="vanilla", choices=sorted(PRESETS))
    ap.add_argument("--model", default="tiny", choices=sorted(MODEL_PRESETS))
    ap.add_argument("--tkw", nargs="*", default=[], help="task kwargs key=value")
    ap.add_argument("--data_dir", default="data/fineweb_edu")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--eval_scales", default="1,2,4")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--evict_eval", action="store_true", help="KV-cache eviction study (token-by-token decoding)")
    return ap.parse_args(argv)


def build_task(args):
    if args.task == "lm":
        return LMData(args.data_dir, seq_len=args.seq_len)
    return make_task(args.task, args.seq_len, Vocab(), **parse_kv(args.tkw))


def build_model_cfg(args, vocab_size):
    kw = dict(MODEL_PRESETS[args.model])
    kw.update(PRESETS[args.variant])
    kw["vocab_size"] = vocab_size
    kw["grad_ckpt"] = args.grad_ckpt
    return ModelConfig(**kw)


def amp_dtype(name, device):
    if device.type != "cuda" or name == "fp32":
        return None
    if name == "auto":
        name = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return torch.bfloat16 if name == "bf16" else torch.float16


def lr_at(step, args):
    warm = max(1, int(args.warmup * args.steps))
    if step < warm:
        return args.lr * (step + 1) / warm
    p = (step - warm) / max(1, args.steps - warm)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, args):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.beta2), fused=fused)


def run_name(args):
    return args.name or f"{args.task}_{args.variant}_{args.model}_L{args.seq_len}_s{args.seed}"


def save_atomic(obj, path):
    tmp = path + ".tmp"
    if path.endswith(".json"):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dt = amp_dtype(args.amp, device)

    task = build_task(args)
    vocab_size = task.vocab_size if args.task == "lm" else Vocab().size
    if args.task == "lm":
        vocab_size = int(math.ceil(vocab_size / 64) * 64)
    cfg = build_model_cfg(args, vocab_size)
    if args.evict_eval and cfg.retire_iters > 1:
        raise ValueError(f"--evict_eval decodes token by token (inference.decode), which implements retire_iters=1 "
                         f"only (got retire_iters={cfg.retire_iters})")
    model = Model(cfg).to(device)
    fwd = torch.compile(model) if args.compile else model
    compile_status = "on" if args.compile else "off"
    opt = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))

    name = run_name(args)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    scales = [int(s) for s in args.eval_scales.split(",")]
    gen = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    cpu_gen = torch.Generator().manual_seed(10_000 + args.seed)
    print(f"[{name}] params(non-emb)={model.num_params() / 1e6:.2f}M amp={dt} device={device}", flush=True)

    def evaluate():
        if args.task == "lm":
            return eval_lm(model, task, scales, args.eval_batches, args.batch, device, dt)
        return eval_task(model, task, scales, args.eval_batches, args.batch, device, dt)

    t_start = time.time()
    tok_seen, t_meas, tok_meas = 0, None, 0
    history = []
    nonfinite = 0
    oom_fix = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        def micro_steps():
            acc, ntok = 0.0, 0
            for _ in range(args.accum):
                if args.task == "lm":
                    d = task.sample(args.batch, gen=cpu_gen, device=device)
                else:
                    d = task.sample(args.batch, gen=gen, device=device)
                with autocast(device, dt):
                    _, loss = fwd(d["input"], d["target"])
                scaler.scale(loss / args.accum).backward()
                acc += float(loss.detach()) / args.accum
                ntok += d["input"].numel()
            return acc, ntok

        for attempt in range(5):
            try:
                loss_acc, ntok = micro_steps()
                break
            except torch.cuda.OutOfMemoryError:
                if step > 5 or args.batch == 1 or attempt == 4:
                    raise
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if not model.cfg.grad_ckpt:
                    model.cfg.grad_ckpt = True
                    oom_fix.append("grad_ckpt")
                else:
                    args.batch, args.accum = args.batch // 2, args.accum * 2
                    oom_fix.append(f"batch={args.batch},accum={args.accum}")
                print(f"OOM at step {step}: retry with {oom_fix[-1]}", flush=True)
            except Exception as e:
                if fwd is model or step > 0 or attempt == 4:
                    raise
                print(f"torch.compile failed ({type(e).__name__}: {str(e)[:300]}); continuing in eager mode",
                      flush=True)
                compile_status = "failed"
                fwd = model
                opt.zero_grad(set_to_none=True)
        tok_seen += ntok
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not torch.isfinite(gnorm):
            nonfinite += 1
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step == 10:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_meas, tok_meas = time.time(), tok_seen
        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": loss_acc, "lr": lr_at(step, args), "gnorm": float(gnorm),
                   "elapsed_s": time.time() - t_start}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            ev = evaluate()
            ev["step"] = step + 1
            history.append(ev)
            print("eval", json.dumps(ev), flush=True)
            log_f.write(json.dumps({"eval": ev}) + "\n")

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t_start
    tok_per_s = (tok_seen - tok_meas) / (time.time() - t_meas) if t_meas else None
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    final = evaluate()
    if args.task == "lm":
        sb = next(task.fixed_batches(min(4, args.batch), 1, device=device))
    else:
        sb = task.sample(min(16, args.batch), gen=torch.Generator(device=device).manual_seed(999), device=device)
    stats = eval_stats(model, sb, device, dt)
    bench = benchmark(model, vocab_size, args.seq_len, min(args.batch, 8), device, dt)

    evict = None
    if args.evict_eval:
        from inference import eval_eviction
        from attention import AbsorbAttention, SelectiveAttention
        N = args.seq_len
        settings = [{"policy": "none"}]
        for frac in (0.5, 0.25):
            b = max(8, int(N * frac))
            settings += [{"policy": "tova", "budget": b}, {"policy": "window", "budget": b}]
        att = model.blocks[0].attn
        if isinstance(att, SelectiveAttention) or (isinstance(att, AbsorbAttention) and cfg.retire):
            for frac in (0.5, 0.25):
                settings.append({"policy": "retire_budget", "budget": max(8, int(N * frac))})
            settings += [{"policy": "retire_tau", "tau": t} for t in (1.0, 2.0, 4.0)]
        evict = eval_eviction(model, task, settings, batch=min(32, args.batch) if args.task != "lm" else 4,
                              n_batches=2, device=str(device), amp_dtype=dt)
        for r in evict:
            print("evict", json.dumps(r), flush=True)

    results = {
        "name": name, "task": args.task, "variant": args.variant, "seed": args.seed,
        "args": vars(args), "model_cfg": cfg.to_dict(),
        "params_nonemb": model.num_params(), "params_total": model.num_params(False),
        "steps_done": step + 1, "train_time_s": train_time, "train_tok_per_s": tok_per_s,
        "train_peak_MiB": peak_mem, "nonfinite_grad_steps": nonfinite, "oom_fix": oom_fix,
        "compile": compile_status, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp_dtype": str(dt).replace("torch.", "") if dt is not None else "fp32",
        "final": final, "history": history, "stats": stats, "bench": bench, "evict": evict,
    }
    if args.save_ckpt:
        save_atomic({"model": model.state_dict(), "model_cfg": cfg.to_dict(), "args": vars(args)},
                    os.path.join(out_dir, "ckpt.pt"))
    save_atomic(results, os.path.join(out_dir, "results.json"))
    print("final", json.dumps(final), flush=True)
    print("stats", json.dumps(stats["summary"]), flush=True)
    log_f.close()
    return results


if __name__ == "__main__":
    main()
