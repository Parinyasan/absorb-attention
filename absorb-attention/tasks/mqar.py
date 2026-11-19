import torch

from .vocab import BOS, Vocab, pack, rand_distinct


class MQAROverwrite:
    name = "mqar"

    def __init__(self, vocab: Vocab, seq_len=256, n_keys=16, writes=4, n_queries=16, extrap="scale"):
        self.v, self.seq_len, self.n_keys, self.writes, self.n_queries, self.extrap = (
            vocab, seq_len, n_keys, writes, n_queries, extrap)

    def sample(self, B, scale=1, gen=None, device="cpu"):
        v = self.v
        s = scale if self.extrap == "scale" else 1
        nk, Q = self.n_keys * s, min(self.n_queries * s, self.n_keys * s)
        P, N = nk * self.writes, self.seq_len * scale
        gap = N - (1 + 2 * P + 2 * Q)
        assert gap >= 0, f"seq_len too short: need {1 + 2 * P + 2 * Q}, have {N}"
        assert nk <= v.n_keys

        kp = rand_distinct(B, v.n_keys, nk, gen, device)
        slot = torch.arange(nk, device=device).repeat(self.writes)
        order = torch.rand(B, P, generator=gen, device=device).argsort(-1)
        slot_k = slot[order]
        vals = torch.randint(0, v.n_vals, (B, P), generator=gen, device=device)
        pos = torch.arange(P, device=device).expand(B, P)
        last = torch.zeros(B, nk, dtype=torch.long, device=device).scatter_reduce(
            1, slot_k, pos, reduce="amax", include_self=False)
        last_val = vals.gather(1, last)

        qk = rand_distinct(B, nk, Q, gen, device)
        kv = torch.stack([kp.gather(1, slot_k) + v.key0, vals + v.val0], -1).view(B, 2 * P)
        q_keys = kp.gather(1, qk) + v.key0
        q_ans = last_val.gather(1, qk) + v.val0
        qa = torch.stack([q_keys, q_ans], -1).view(B, 2 * Q)
        noise = torch.randint(v.noise0, v.noise0 + v.n_noise, (B, gap), generator=gen, device=device)
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        seq = torch.cat([bos, kv, noise, qa], dim=1)

        target = torch.full_like(seq, -100)
        qpos = 1 + 2 * P + gap + 2 * torch.arange(Q, device=device)
        target[:, qpos] = q_ans
        return pack(seq, target)
