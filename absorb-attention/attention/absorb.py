import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import AttentionBase, block_masks, carry_cumsum, gate_linear, softmax_with_null, up


def share(S, mode):
    if mode == "mean":
        return S.mean(1, keepdim=True)
    if mode == "head0":
        return S[:, :1]
    return S


class AbsorbAttention(AttentionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        D, H = cfg.d_model, cfg.n_head
        if cfg.null_mode == "query":
            self.null_proj = gate_linear(D, H, cfg.null_bias)
        elif cfg.null_mode == "const":
            self.null_logit = nn.Parameter(torch.full((H,), float(cfg.null_bias)))
        elif cfg.null_mode != "none":
            raise ValueError(cfg.null_mode)
        if cfg.retire and cfg.retire_src in ("attn", "odds", "attn_scaled"):
            self.rho_proj = gate_linear(D, H, cfg.rho_bias)
        if cfg.out_gate in ("elem", "head"):
            self.out_gate = nn.Linear(D, D if cfg.out_gate == "elem" else H, bias=False)
        elif cfg.out_gate != "none":
            raise ValueError(cfg.out_gate)
        if cfg.read_gate:
            self.gate_x = gate_linear(D, H, cfg.read_gate_bias)
            self.gate_o = nn.Parameter(torch.zeros(H, self.d_head))

    def out_gate_values(self, h):
        B, N, _ = h.shape
        g = torch.sigmoid(self.out_gate(h))
        if self.cfg.out_gate == "head":
            return g.view(B, N, self.n_head, 1).transpose(1, 2)
        return g.view(B, N, self.n_head, self.d_head).transpose(1, 2)

    def null_logits(self, h):
        if self.cfg.null_mode == "query":
            return up(self.null_proj(h)).transpose(1, 2)
        if self.cfg.null_mode == "const":
            return up(self.null_logit).view(1, -1, 1)
        return None

    def forward(self, h, ctx):
        cfg = self.cfg
        q, k, v = self.heads(h)
        s_full = self.null_logits(h)
        rho_full = (F.softplus(up(self.rho_proj(h))).transpose(1, 2)
                    if cfg.retire and cfg.retire_src in ("attn", "odds", "attn_scaled") else None)
        bound_src = cfg.bind and cfg.bind_retire and s_full is not None
        if cfg.retire_iters > 1 and getattr(ctx, "chunk", 0) and not ctx.record:
            raise NotImplementedError("retire_iters > 1 needs the full (unchunked) computation")

        def fn(lg, r0, r1, carry):
            s = None if s_full is None else (s_full[..., r0:r1] if s_full.shape[-1] > 1 else s_full)
            rho = None if rho_full is None else rho_full[..., r0:r1]

            def read(x):
                if s is None:
                    return torch.softmax(x, dim=-1), None
                return softmax_with_null(x, s)

            def source(x):
                if bound_src:
                    return softmax_with_null(x, s)[0]
                return torch.softmax(x, dim=-1)

            R = None
            if cfg.retire:
                keep = block_masks(r0, r1, lg.device, cfg.retire_bos)[1]
                if cfg.retire_src == "relu":
                    S = F.relu(lg)
                elif cfg.retire_src == "odds":
                    margin = lg - s.unsqueeze(-1) if bound_src else lg
                    S = rho.unsqueeze(-1) * F.relu(margin)
                elif cfg.retire_src == "attn_scaled":
                    n_keys = torch.arange(r0 + 1, r1 + 1, device=lg.device, dtype=lg.dtype)[:, None]
                    S = rho.unsqueeze(-1) * source(lg) * n_keys
                else:
                    S = rho.unsqueeze(-1) * source(lg)
                R, new_carry = carry_cumsum(share(S.masked_fill(~keep, 0.0), cfg.retire_share), carry)
                a, n = read(lg - R)
                for _ in range(cfg.retire_iters - 1 if cfg.retire_src == "attn" else 0):
                    a_src = a if bound_src else source(lg - R)
                    R, new_carry = carry_cumsum(share((rho.unsqueeze(-1) * a_src).masked_fill(~keep, 0.0),
                                                      cfg.retire_share), None)
                    a, n = read(lg - R)
                carry = new_carry
            else:
                a, n = read(lg)
            return a, n, carry, {"R": R}

        o, n, last = self.blockwise(q, k, v, ctx, fn)
        if cfg.read_gate:
            g = up(self.gate_x(h)).transpose(1, 2) + torch.einsum("bhnd,hd->bhn", up(o), up(self.gate_o))
            o = o * torch.sigmoid(g).unsqueeze(-1).to(o.dtype)
        if cfg.out_gate != "none":
            o = o * self.out_gate_values(h).to(o.dtype)

        aux = {}
        if n is not None:
            aux["null"] = n
        if ctx.record:
            aux["attn"] = last["attn"]
            if last.get("R") is not None:
                aux["R"] = last["R"].expand_as(last["attn"])
            if rho_full is not None:
                aux["rho"] = rho_full
        return self.merge(o), aux
