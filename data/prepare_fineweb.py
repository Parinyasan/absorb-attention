import argparse
import json
import os
import sys

import numpy as np


def verify(out):
    try:
        with open(os.path.join(out, "meta.json")) as f:
            counts = json.load(f)["counts"]
        bad = [s for s in ("train", "val") if os.path.getsize(os.path.join(out, f"{s}.bin")) != 2 * counts[s]]
    except (OSError, KeyError, ValueError) as e:
        print(f"[data] {out}: incomplete ({e.__class__.__name__}: {e})", flush=True)
        return False
    if bad:
        print(f"[data] {out}: size mismatch in {bad}", flush=True)
        return False
    print(f"[data] {out}: OK, train {counts['train'] / 1e9:.3f}B tokens, val {counts['val'] / 1e6:.1f}M tokens",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=2.1e9)
    ap.add_argument("--val_tokens", type=float, default=1e7)
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--batch_docs", type=int, default=1000)
    ap.add_argument("--verify", action="store_true", help="only check an existing output directory")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.out) else 1)

    token = os.environ.get("HF_TOKEN") or None
    print("HF_TOKEN found: authenticated Hub access" if token else "no HF_TOKEN: anonymous Hub access", flush=True)

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2", token=token)
    eot = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True, token=token)

    targets = {"val": int(a.val_tokens), "train": int(a.tokens)}
    files = {s: open(os.path.join(a.out, f"{s}.bin"), "wb") for s in targets}
    counts = {s: 0 for s in targets}
    split = "val"
    buf = []

    def flush(texts):
        nonlocal split
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        for doc in ids:
            if split is None:
                return
            arr = np.array(doc + [eot], dtype=np.uint16)
            files[split].write(arr.tobytes())
            counts[split] += len(arr)
            if counts[split] >= targets[split]:
                split = "train" if split == "val" else None

    for ex in ds:
        buf.append(ex["text"])
        if len(buf) >= a.batch_docs:
            flush(buf)
            buf = []
            print({k: f"{v / 1e6:.1f}M" for k, v in counts.items()}, flush=True)
            if split is None:
                break
    if buf and split is not None:
        flush(buf)
    for f in files.values():
        f.close()

    punct = sorted({i for s in [".", ",", ";", ":", "!", "?", "\n", "\n\n", " .", " ,", " ;", " :", " -", "-", ")", "(", '"', ' "']
                    for i in tok.encode(s) if len(tok.encode(s)) == 1})
    meta = {"vocab_size": len(tok), "eot": eot, "punct_ids": punct, "tokenizer": "gpt2", "counts": counts,
            "dataset": f"{a.dataset}/{a.subset}"}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done", meta, flush=True)
    ok = verify(a.out)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
