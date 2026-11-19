import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import AttentionBase, block_masks, carry_cumsum, softmax_with_null, up


class GatedAttention(AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.headwise = cfg.gate_headwise
        self.gate = nn.Linear(cfg.d_model, cfg.n_head if self.headwise else cfg.d_model, bias=False)

    def gate_values(self, h):
        B, N, _ = h.shape
        g = torch.sigmoid(self.gate(h))
        if self.headwise:
            return g.view(B, N, self.n_head, 1).transpose(1, 2)
        return g.view(B, N, self.n_head, self.d_head).transpose(1, 2)

    def forward(self, h, ctx):
        q, k, v = self.heads(h)
        if ctx.record or not ctx.fast:
            a = torch.softmax(self.logits(q, k), dim=-1)
            o = self.mix(a, v)
        else:
            a, o = None, self.sdpa(q, k, v)
        g = self.gate_values(h)
        o = o * g.to(o.dtype)
        aux = {"attn": a, "gate": g.float().mean(-1)} if ctx.record else {}
        return self.merge(o), aux


class SinkAttention(AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sink = nn.Parameter(torch.full((cfg.n_head,), float(cfg.sink_init)))

    def forward(self, h, ctx):
        q, k, v = self.heads(h)
        sink = up(self.sink).view(1, -1, 1)

        def fn(lg, r0, r1, carry):
            a, n = softmax_with_null(lg, sink)
            return a, n, None, {}

        o, n, last = self.blockwise(q, k, v, ctx, fn)
        aux = {"null": n}
        if ctx.record:
            aux["attn"] = last["attn"]
        return self.merge(o), aux


class SelectiveAttention(AttentionBase):
    def forward(self, h, ctx):
        q, k, v = self.heads(h)

        def fn(lg, r0, r1, carry):
            keep = block_masks(r0, r1, lg.device, self.cfg.retire_bos)[1]
            S = F.relu(lg[:, :1]).masked_fill(~keep, 0.0)
            Fm, carry = carry_cumsum(S, carry)
            return torch.softmax(lg - Fm, dim=-1), None, carry, {"R": Fm.expand_as(lg)}

        o, _, last = self.blockwise(q, k, v, ctx, fn)
        aux = {"attn": last["attn"], "R": last["R"]} if ctx.record else {}
        return self.merge(o), aux


class ForgettingAttention(AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fgate = nn.Linear(cfg.d_model, cfg.n_head, bias=True)
        self.fgate.bias_init = float(cfg.fox_bias)

    def forward(self, h, ctx):
        q, k, v = self.heads(h)
        logf = F.logsigmoid(up(self.fgate(h))).transpose(1, 2)
        c = logf.cumsum(-1)

        def fn(lg, r0, r1, carry):
            decay = c[..., r0:r1, None] - c[..., None, :r1]
            decay = decay.masked_fill(~block_masks(r0, r1, lg.device)[0], 0.0)
            return torch.softmax(lg + decay, dim=-1), None, None, {}

        o, _, last = self.blockwise(q, k, v, ctx, fn)
        aux = {"attn": last["attn"], "fgate": logf.exp()} if ctx.record else {}
        return self.merge(o), aux


class StickBreakingAttention(AttentionBase):
    def forward(self, h, ctx):
        q, k, v = self.heads(h)

        def fn(lg, r0, r1, carry):
            cm = block_masks(r0, r1, lg.device)[0]
            z = lg.masked_fill(~cm, 0.0)
            log_beta = F.logsigmoid(z)
            log_1m = F.logsigmoid(-z).masked_fill(~cm, 0.0)
            U = log_1m.flip(-1).cumsum(-1).flip(-1) - log_1m
            a = torch.exp(log_beta + U).masked_fill(~cm, 0.0)
            return a, (1.0 - a.sum(-1)).clamp_min(0.0), None, {}

        o, n, last = self.blockwise(q, k, v, ctx, fn)
        aux = {"null": n}
        if ctx.record:
            aux["attn"] = last["attn"]
        return self.merge(o), aux
