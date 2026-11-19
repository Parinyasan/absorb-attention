import torch

from .vocab import BOS, EQ, QRY, SEP, Vocab, pack, rand_distinct


class VariableTracking:
    name = "vt"

    def __init__(self, vocab: Vocab, seq_len=256, n_vars=8, p_assign=0.4, p_query=0.2, scale_vars=False):
        self.v, self.seq_len, self.n_vars = vocab, seq_len, n_vars
        self.p = torch.tensor([1.0 - p_assign - p_query, p_assign, p_query])
        self.scale_vars = scale_vars

    def sample(self, B, scale=1, gen=None, device="cpu"):
        v = self.v
        N = self.seq_len * scale
        M = (N - 1) // 4
        nv = self.n_vars * (scale if self.scale_vars else 1)
        kp = rand_distinct(B, v.n_keys, nv, gen, device)
        types = torch.multinomial(self.p.to(device).expand(B, 3), M, replacement=True, generator=gen)
        var = torch.randint(0, nv, (B, M), generator=gen, device=device)
        val = torch.randint(0, v.n_vals, (B, M), generator=gen, device=device)

        is_a = types == 1
        idx = torch.arange(M, device=device)
        earlier = idx[None, :] < idx[:, None]
        same = (var[:, :, None] == var[:, None, :]) & is_a[:, None, :] & earlier
        last = torch.where(same, idx.view(1, 1, M), torch.full_like(idx, -1).view(1, 1, M)).amax(-1)
        q_ok = (types == 2) & (last >= 0)
        types = torch.where((types == 2) & ~q_ok, torch.zeros_like(types), types)
        ans = val.gather(1, last.clamp_min(0))

        name = kp.gather(1, var) + v.key0
        noise = torch.randint(v.noise0, v.noise0 + v.n_noise, (B, M, 4), generator=gen, device=device)
        c = lambda t: torch.full((B, M), t, dtype=torch.long, device=device)
        assign = torch.stack([name, c(EQ), val + v.val0, c(SEP)], -1)
        query = torch.stack([c(QRY), name, ans + v.val0, c(SEP)], -1)
        t = types.unsqueeze(-1)
        slots = torch.where(t == 1, assign, torch.where(t == 2, query, noise))

        tgt = torch.full((B, M, 4), -100, dtype=torch.long, device=device)
        tgt[..., 1] = torch.where(q_ok, ans + v.val0, torch.full_like(ans, -100))

        tail = N - 1 - 4 * M
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        pad = torch.randint(v.noise0, v.noise0 + v.n_noise, (B, tail), generator=gen, device=device)
        seq = torch.cat([bos, slots.view(B, 4 * M), pad], 1)
        target = torch.cat([torch.full((B, 1), -100, device=device, dtype=torch.long), tgt.view(B, 4 * M),
                            torch.full((B, tail), -100, device=device, dtype=torch.long)], 1)
        return pack(seq, target)
