import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def up(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def causal_mask(n, device):
    return torch.ones(n, n, dtype=torch.bool, device=device).tril_()


def block_masks(r0, r1, device, retire_bos=False):
    rows = torch.arange(r0, r1, device=device)[:, None]
    cols = torch.arange(r1, device=device)[None, :]
    keep = cols < rows
    if not retire_bos:
        keep = keep & (cols > 0)
    return cols <= rows, keep


def carry_cumsum(S, carry):
    R = S.cumsum(dim=-2) - S
    tot = S.sum(dim=-2)
    if carry is not None:
        kc = carry.shape[-1]
        R[..., :kc] = R[..., :kc] + carry.unsqueeze(-2)
        tot[..., :kc] = tot[..., :kc] + carry
    return R, tot


def softmax_with_null(logits, null_logit):
    lse = torch.logaddexp(torch.logsumexp(logits, dim=-1), null_logit)
    a = torch.exp(logits - lse.unsqueeze(-1))
    n = torch.exp(null_logit - lse)
    return a, n


def gate_linear(d_in, d_out, bias):
    lin = nn.Linear(d_in, d_out)
    lin.zero_init = True
    lin.bias_init = float(bias)
    nn.init.zeros_(lin.weight)
    nn.init.constant_(lin.bias, float(bias))
    return lin


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = up(x)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * up(self.weight)).to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, n, device):
        t = torch.arange(n, device=device, dtype=torch.float32)
        f = torch.outer(t, self.inv_freq.to(device))
        return f.cos(), f.sin()


def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = up(x[..., :d]), up(x[..., d:])
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class AttentionBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o.is_residual_proj = True
        self.rotary = Rotary(self.d_head, cfg.rope_base) if cfg.rope else None
        if cfg.qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.d_head), RMSNorm(self.d_head)

    def project(self, h):
        B, N, D = h.shape
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def heads(self, h):
        q, k, v = self.project(h)
        N = h.shape[1]
        if self.rotary is not None:
            cos, sin = self.rotary(N, h.device)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        return q, k, v

    def logits(self, q, k):
        n = q.shape[-2]
        lg = up(torch.matmul(q, k.transpose(-1, -2))) * self.scale
        return lg.masked_fill(~causal_mask(n, q.device), NEG_INF)

    def blockwise(self, q, k, v, ctx, fn):
        N = q.shape[-2]
        chunk = ctx.chunk if (getattr(ctx, "chunk", 0) and not ctx.record) else N
        outs, nulls, carry, last = [], [], None, {}
        for r0 in range(0, N, chunk):
            r1 = min(N, r0 + chunk)
            lg = up(torch.matmul(q[:, :, r0:r1], k[:, :, :r1].transpose(-1, -2))) * self.scale
            lg = lg.masked_fill(~block_masks(r0, r1, q.device)[0], NEG_INF)
            a, n, carry, last = fn(lg, r0, r1, carry)
            outs.append(self.mix(a, v[:, :, :r1]))
            if n is not None:
                nulls.append(n)
            last = dict(last, attn=a)
        return torch.cat(outs, dim=2), (torch.cat(nulls, dim=-1) if nulls else None), last

    def merge(self, o):
        B, H, N, dh = o.shape
        return self.o(o.transpose(1, 2).reshape(B, N, H * dh))

    def mix(self, a, v):
        return torch.matmul(a.to(v.dtype), v)

    def sdpa(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
