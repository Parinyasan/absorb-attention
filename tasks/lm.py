import json
import os

import numpy as np
import torch


class LMData:
    name = "lm"

    def __init__(self, data_dir, seq_len=1024):
        self.seq_len = seq_len
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.splits = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                       for s in ("train", "val")}
        self.punct = torch.tensor(self.meta["punct_ids"], dtype=torch.long)

    @property
    def vocab_size(self):
        return self.meta["vocab_size"]

    def _batch(self, data, starts, N, device):
        x = torch.from_numpy(np.stack([data[i:i + N + 1].astype(np.int64) for i in starts]))
        x = x.to(device, non_blocking=True)
        inp, tgt = x[:, :-1], x[:, 1:]
        return {"input": inp, "target": tgt.contiguous(), "special": torch.isin(inp, self.punct.to(device))}

    def sample(self, B, scale=1, gen=None, device="cpu", split="train"):
        data, N = self.splits[split], self.seq_len * scale
        hi = len(data) - N - 2
        starts = torch.randint(0, hi, (B,), generator=gen).tolist()
        return self._batch(data, starts, N, device)

    def fixed_batches(self, B, n_batches, scale=1, device="cpu", split="val"):
        data, N = self.splits[split], self.seq_len * scale
        stride = N + 1
        total = min(n_batches * B, (len(data) - 1) // stride)
        starts = [i * stride for i in range(total)]
        for b in range(0, total, B):
            yield self._batch(data, starts[b:b + B], N, device)
