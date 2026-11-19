import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import benchmark
from model import Model, ModelConfig
from presets import PRESETS
from train import MODEL_PRESETS, amp_dtype


def compiled(m, a, dev, dt):
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    try:
        cm = torch.compile(m)
        x = torch.randint(0, a.vocab, (2, a.seq_len), device=dev, generator=torch.Generator(dev).manual_seed(0))
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            ref = float(m(x, x)[1])
            got = float(cm(x, x)[1])
        m.train()
        r = benchmark(cm, a.vocab, a.seq_len, a.batch, dev, dt, steps=10)
        return {f"compile_{k}": v for k, v in r.items()} | {"compile_loss_absdiff": abs(ref - got)}
    except torch.cuda.OutOfMemoryError:
        return {"compile_oom": True}
    except Exception as e:
        return {"compile_error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lm100m")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--variants", default="vanilla,gated,selective,selective_wide,sa_null_delta,sa_g1_head,"
                                           "sa_g1,absorb_g1")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="also time a torch.compile'd copy and check its loss matches eager")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    dt = amp_dtype("auto", dev)
    rows = []
    for v in a.variants.split(","):
        cfg = ModelConfig(**{**MODEL_PRESETS[a.model], **PRESETS[v], "vocab_size": a.vocab, "grad_ckpt": a.grad_ckpt})
        m = Model(cfg).to(dev)
        try:
            r = benchmark(m, a.vocab, a.seq_len, a.batch, dev, dt)
        except torch.cuda.OutOfMemoryError:
            r = {"oom": True}
        if a.compile:
            r.update(compiled(m, a, dev, dt))
        r.update(variant=v, params_M=m.num_params() / 1e6)
        rows.append(r)
        print(json.dumps(r), flush=True)
        del m
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(0), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
