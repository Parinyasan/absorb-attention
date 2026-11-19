import torch

from .common import AttentionBase


class VanillaAttention(AttentionBase):
    def forward(self, h, ctx):
        q, k, v = self.heads(h)
        if ctx.record or not ctx.fast:
            a = torch.softmax(self.logits(q, k), dim=-1)
            o = self.mix(a, v)
            aux = {"attn": a} if ctx.record else {}
        else:
            o = self.sdpa(q, k, v)
            aux = {}
        return self.merge(o), aux
