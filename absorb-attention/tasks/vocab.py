from dataclasses import dataclass

import torch

PAD, BOS, EQ, SEP, QRY, DOT, COLON = 0, 1, 2, 3, 4, 5, 6
PUNCT = (EQ, SEP, QRY, DOT, COLON)
N_SPECIAL = 8


@dataclass
class Vocab:
    n_keys: int = 256
    n_vals: int = 128
    n_noise: int = 128

    @property
    def key0(self):
        return N_SPECIAL

    @property
    def val0(self):
        return N_SPECIAL + self.n_keys

    @property
    def noise0(self):
        return N_SPECIAL + self.n_keys + self.n_vals

    @property
    def size(self):
        return N_SPECIAL + self.n_keys + self.n_vals + self.n_noise


def special_mask(tokens):
    m = torch.zeros_like(tokens, dtype=torch.bool)
    for t in PUNCT:
        m |= tokens == t
    return m


def rand_distinct(B, pool, k, gen, device):
    return torch.rand(B, pool, generator=gen, device=device).argsort(-1)[:, :k]


def pack(seq, target):
    return {"input": seq, "target": target, "special": special_mask(seq)}
