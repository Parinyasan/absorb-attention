import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from attention import make_attention
from attention.common import RMSNorm, gate_linear, up


@dataclass
class ModelConfig:
    vocab_size: int = 512
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    mlp_mult: float = 8 / 3
    rope: bool = True
    rope_base: float = 10000.0
    qk_norm: bool = True
    attn: str = "vanilla"
    residual: str = "add"
    ddl_mlp: bool = True
    null_mode: str = "none"
    null_bias: float = -4.0
    retire: bool = False
    rho_bias: float = -6.0
    retire_iters: int = 1
    bind: bool = True
    bind_beta: bool = True
    bind_retire: bool = True
    retire_share: str = "head"
    retire_src: str = "attn"
    retire_bos: bool = False
    out_gate: str = "none"
    beta_bias: float = -4.0
    read_gate: bool = False
    read_gate_bias: float = 4.0
    sink_init: float = -4.0
    gate_headwise: bool = False
    fox_bias: float = 0.0
    ddl_beta_bias: float = 0.0
    ddl_eps: float = 1e-5
    delta_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    grad_ckpt: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.retire_iters < 1:
            raise ValueError(f"retire_iters={self.retire_iters} must be >= 1")
        if self.retire_iters > 1 and not (self.attn == "absorb" and self.retire and self.retire_src == "attn"):
            raise ValueError(f"retire_iters={self.retire_iters} needs attn='absorb', retire=True, retire_src='attn' "
                             f"(got attn={self.attn!r}, retire={self.retire}, retire_src={self.retire_src!r})")
        if self.out_gate != "none" and self.attn != "absorb":
            raise ValueError(f"out_gate={self.out_gate!r} is only applied by attn='absorb' (got attn={self.attn!r})")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass
class Ctx:
    record: bool = False
    fast: bool = True
    special: Optional[torch.Tensor] = None
    chunk: int = 0
    stats: list = field(default_factory=list)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(math.ceil(cfg.mlp_mult * cfg.d_model / 64) * 64)
        self.up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.down.is_residual_proj = True

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class DeltaWrite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bind = cfg.bind and cfg.bind_beta
        self.eps = cfg.delta_eps
        self.proj = gate_linear(cfg.d_model, 1, cfg.beta_bias)

    def forward(self, x, h, m, null):
        gate = torch.sigmoid(up(self.proj(h)))
        beta = gate * (1.0 - null.mean(1)).unsqueeze(-1) if (self.bind and null is not None) else gate
        xf, mf = up(x), up(m)
        u = mf * torch.rsqrt(mf.pow(2).sum(-1, keepdim=True) + self.eps ** 2)
        out = xf - beta * (u * xf).sum(-1, keepdim=True) * u + mf
        return out.to(x.dtype), beta, gate


class GainWrite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = gate_linear(cfg.d_model, 1, 0.0)

    def forward(self, x, h, m):
        g = 2.0 * torch.sigmoid(up(self.proj(h)))
        return (up(x) + g * up(m)).to(x.dtype), g


class DDLUpdate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg.ddl_eps
        self.beta = gate_linear(cfg.d_model, 1, cfg.ddl_beta_bias)
        self.beta.zero_init = False
        self.v_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, x, h, m):
        beta = 2.0 * torch.sigmoid(up(self.beta(h)))
        v = 4.0 * torch.sigmoid(up(self.v_proj(x)))
        xf, mf = up(x), up(m)
        k = mf * torch.rsqrt(mf.pow(2).sum(-1, keepdim=True) + self.eps ** 2)
        out = xf + beta * (v - (k * xf).sum(-1, keepdim=True)) * k
        return out.to(x.dtype), beta


class Block(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.cfg, self.idx = cfg, idx
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = make_attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        if cfg.residual == "delta":
            self.delta = DeltaWrite(cfg)
        elif cfg.residual == "gain":
            self.gain = GainWrite(cfg)
        elif cfg.residual == "ddl":
            self.ddl = DDLUpdate(cfg)
            if cfg.ddl_mlp:
                self.ddl2 = DDLUpdate(cfg)
        elif cfg.residual != "add":
            raise ValueError(cfg.residual)

    def forward(self, x, ctx):
        h = self.n1(x)
        m, aux = self.attn(h, ctx)
        beta = gate = None
        if self.cfg.residual == "delta":
            x, beta, gate = self.delta(x, h, m, aux.get("null"))
        elif self.cfg.residual == "gain":
            x, beta = self.gain(x, h, m)
        elif self.cfg.residual == "ddl":
            x, beta = self.ddl(x, h, m)
        else:
            x = x + m
        h2 = self.n2(x)
        f = self.mlp(h2)
        if self.cfg.residual == "ddl" and self.cfg.ddl_mlp:
            x, _ = self.ddl2(x, h2, f)
        else:
            x = x + f
        if ctx.record:
            ctx.stats.append(layer_stats(aux, beta, gate, x, ctx.special, self.cfg.retire_bos))
        return x


@torch.no_grad()
def layer_stats(aux, beta, gate, x, special, retire_bos=False):
    s = {}
    if "null" in aux:
        s["null_mean_per_head"] = aux["null"].float().mean(dim=(0, 2)).tolist()
        s["null_mean"] = float(aux["null"].float().mean())
    if beta is not None:
        s["beta_mean"] = float(beta.mean())
        s["gate_mean"] = float(gate.mean()) if gate is not None else float(beta.mean())
    a = aux.get("attn")
    if a is not None:
        a = a.float()
        s["first_mass"] = float(a[:, :, 1:, 0].mean())
        if special is not None:
            sp = special.to(a.dtype)[:, None, None, :]
            s["punct_mass"] = float((a * sp).sum(-1)[:, :, 1:].mean())
            cnt = special.float().cumsum(-1) / torch.arange(1, special.shape[1] + 1, device=special.device)
            s["punct_uniform"] = float(cnt[:, 1:].mean())
        ent = -(a.clamp_min(1e-12).log() * a).sum(-1)
        s["attn_entropy"] = float(ent[:, :, 1:].mean())
    R = aux.get("R")
    if R is not None:
        n = R.shape[-1]
        valid = torch.ones(n, n, dtype=torch.bool, device=R.device).tril_(-2)
        valid[:, 0] = False
        if valid.any():
            r = R.float()[..., valid]
            s["R_mean"] = float(r.mean())
            s["retired_frac_R>2"] = float((r > 2.0).float().mean())
        if retire_bos and n > 2:
            r0 = R.float()[..., 2:, 0]
            s["R_bos_mean"] = float(r0.mean())
            s["retired_frac_R_bos>2"] = float((r0 > 2.0).float().mean())
    if "rho" in aux:
        s["rho_mean"] = float(aux["rho"].float().mean())
    xa = x.float().abs()
    s["max_act"] = float(xa.max())
    s["median_act"] = float(xa.median())
    return s


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight

    def reset_parameters(self):
        std = self.cfg.init_std
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                if getattr(mod, "zero_init", False):
                    nn.init.zeros_(mod.weight)
                else:
                    s = std / math.sqrt(2 * self.cfg.n_layer) if getattr(mod, "is_residual_proj", False) else std
                    nn.init.normal_(mod.weight, 0.0, s)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, getattr(mod, "bias_init", 0.0))
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, 0.0, std)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, ctx=None):
        ctx = ctx or Ctx()
        x = self.emb(idx)
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training and not ctx.record:
                x = checkpoint(blk, x, ctx, use_reentrant=False)
            else:
                x = blk(x, ctx)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(up(logits).view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss
