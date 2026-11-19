import argparse
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import Ctx, Model, ModelConfig

TASKS = "lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,sciq,openbookqa"


def harness_version():
    from importlib import metadata
    for name in ("lm_eval", "lm-eval"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def prev_path(out_path, tag="prev"):
    root, ext = os.path.splitext(out_path)
    return f"{root}_{tag}{ext}"


def dump_atomic(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_result(res, out_path):
    if not res["tasks"] and os.path.exists(out_path):
        path = prev_path(out_path, "failed")
        dump_atomic(res, path)
        return path
    prev = prev_path(out_path)
    if os.path.exists(out_path) and not os.path.exists(prev):
        tmp = f"{prev}.tmp{os.getpid()}"
        shutil.copyfile(out_path, tmp)
        os.replace(tmp, prev)
    dump_atomic(res, out_path)
    return out_path


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{**ck["model_cfg"], "grad_ckpt": False})
    model = Model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def score(model, pairs, eot, max_len, batch, device, dtype, chunk=0):
    order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
    out = [None] * len(pairs)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        seqs = [([eot] + pairs[i][0] + pairs[i][1])[-(max_len + 1):] for i in idx]
        width = max(len(s) for s in seqs) - 1
        inp = torch.full((len(idx), width), eot, dtype=torch.long)
        for r, s in enumerate(seqs):
            inp[r, :len(s) - 1] = torch.tensor(s[:-1])
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            logits, _ = model(inp.to(device), ctx=Ctx(chunk=chunk))
        logp = F.log_softmax(logits.float(), dim=-1)
        for r, i in enumerate(idx):
            s, n = seqs[r], len(pairs[i][1])
            start = len(s) - 1 - n
            tgt = torch.tensor(s[-n:], device=device)
            lp = logp[r, start:start + n]
            out[i] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
    return out


def encode_pair(tok, context, continuation):
    n_sp = len(context) - len(context.rstrip())
    if n_sp:
        continuation, context = context[-n_sp:] + continuation, context[:-n_sp]
    whole = tok.encode(context + continuation)
    ctx = tok.encode(context) if context else []
    return ctx, whole[len(ctx):]


def make_lm(model, tok, max_len, batch, device, dtype):
    from lm_eval.api.model import LM

    class CheckpointLM(LM):
        def loglikelihood(self, requests, disable_tqdm=False):
            pairs = [encode_pair(tok, *r.args) for r in requests]
            return score(model, pairs, tok.eos_token_id, max_len, batch, device, dtype)

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("perplexity tasks are measured by train.py/eval.py")

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError("only log-likelihood (multiple-choice / lambada) tasks are used")

    return CheckpointLM()


def prefetch(tasks):
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    for t in tasks:
        try:
            get_task_dict([t], tm)
            print(f"[prefetch] {t}: ok", flush=True)
        except Exception as e:
            print(f"[prefetch] {t}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (quick checks only)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None, help="default: zeroshot.json next to the checkpoint")
    ap.add_argument("--prefetch", action="store_true", help="only download/prepare the task datasets")
    ap.add_argument("--require_lm_eval", default=os.environ.get("LM_EVAL_VERSION") or None,
                    help="refuse to run unless this lm-eval version is installed (default $LM_EVAL_VERSION)")
    a = ap.parse_args()
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    tasks = a.tasks.split(",")
    if a.prefetch:
        prefetch(tasks)
        return
    if not a.ckpt:
        ap.error("--ckpt is required")
    have = harness_version()
    if a.require_lm_eval and have != a.require_lm_eval:
        sys.exit(f"lm-eval {have or '(not installed)'} is installed but {a.require_lm_eval} is required: not scoring "
                 f"{a.ckpt}")

    import lm_eval
    import transformers
    from transformers import GPT2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model, ck = load_model(a.ckpt, device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2", token=os.environ.get("HF_TOKEN") or None)
    max_len = ck["args"].get("seq_len", 1024)
    lm = make_lm(model, tok, max_len, a.batch, device, dtype)

    out_path = a.out or os.path.join(os.path.dirname(a.ckpt), "zeroshot.json")
    res = {"ckpt": a.ckpt, "variant": ck["args"].get("variant"), "seed": ck["args"].get("seed"),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "lm_eval": have or getattr(lm_eval, "__version__", None), "torch": torch.__version__,
           "transformers": getattr(transformers, "__version__", None),
           "limit": a.limit, "tasks": {}, "errors": {}}
    for t in tasks:
        t0 = time.time()
        try:
            r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=0, limit=a.limit, bootstrap_iters=1000,
                                        log_samples=False)
            metrics = {k.split(",")[0]: v for k, v in r["results"][t].items()
                       if isinstance(v, (int, float)) and not k.startswith("alias")}
            res["tasks"][t] = metrics
            print(f"[{t}] {json.dumps(metrics)} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            res["errors"][t] = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"[{t}] FAILED {res['errors'][t]}", flush=True)
    path = write_result(res, out_path)
    print(f"wrote {path} (lm-eval {res['lm_eval']}, torch {res['torch']})"
          + ("" if path == out_path else f"; no task scored, {out_path} left as it was"), flush=True)
    sys.exit(1 if res["errors"] and not res["tasks"] else 0)


if __name__ == "__main__":
    main()
