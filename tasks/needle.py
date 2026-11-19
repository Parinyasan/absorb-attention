import torch

from .vocab import BOS, COLON, DOT, QRY, Vocab, pack, rand_distinct


class Needle:
    name = "niah"

    def __init__(self, vocab: Vocab, seq_len=256, n_needles=4, n_queries=2, p_dot=0.1):
        self.v, self.seq_len, self.n_needles, self.n_queries, self.p_dot = vocab, seq_len, n_needles, n_queries, p_dot

    def sample(self, B, scale=1, gen=None, device="cpu"):
        v, K, Q = self.v, self.n_needles, self.n_queries
        N = self.seq_len * scale
        Hn = N - 1 - 3 * Q
        seg = Hn // K
        assert seg >= 4
        hay = torch.randint(v.noise0, v.noise0 + v.n_noise, (B, Hn), generator=gen, device=device)
        hay = torch.where(torch.rand(B, Hn, generator=gen, device=device) < self.p_dot, torch.full_like(hay, DOT), hay)

        keys = rand_distinct(B, v.n_keys, K, gen, device) + v.key0
        vals = torch.randint(0, v.n_vals, (B, K), generator=gen, device=device) + v.val0
        off = (torch.rand(B, K, generator=gen, device=device) * (seg - 3)).long()
        start = torch.arange(K, device=device) * seg + off
        cols = start.unsqueeze(-1) + torch.arange(4, device=device)
        toks = torch.stack([torch.full_like(keys, COLON), keys, vals, torch.full_like(keys, DOT)], -1)
        hay = hay.scatter(1, cols.view(B, -1), toks.view(B, -1))

        qi = rand_distinct(B, K, Q, gen, device)
        qk, qv = keys.gather(1, qi), vals.gather(1, qi)
        qa = torch.stack([torch.full_like(qk, QRY), qk, qv], -1).view(B, 3 * Q)
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        seq = torch.cat([bos, hay, qa], 1)
        target = torch.full_like(seq, -100)
        target[:, 1 + Hn + 1 + 3 * torch.arange(Q, device=device)] = qv
        return pack(seq, target)
