import math

import torch
import torch.nn.functional as F

from attention import AbsorbAttention, ForgettingAttention, GatedAttention, SelectiveAttention, SinkAttention, \
    StickBreakingAttention
from attention.common import NEG_INF, apply_rotary, up


class LayerCache:
    def __init__(self, B, H, N, dh, device, dtype, raw=False):
        self.k = torch.zeros(B, H, N, dh, device=device, dtype=dtype)
        self.v = torch.zeros(B, H, N, dh, device=device, dtype=dtype)
        self.R = torch.zeros(B, H, N, device=device, dtype=dtype)
        self.alive = torch.zeros(B, H, N, device=device, dtype=torch.bool)
        self.logf = torch.zeros(B, H, N, device=device, dtype=dtype)
        self.kr = torch.zeros_like(self.k) if raw else None


def _null_logit(att, h):
    if isinstance(att, SinkAttention):
        return up(att.sink).view(1, -1)
    if isinstance(att, AbsorbAttention):
        s = att.null_logits(h)
        return None if s is None else s.reshape(s.shape[0], -1)
    return None


def _read(att, eff, s, cm):
    if isinstance(att, StickBreakingAttention):
        z = eff.masked_fill(~cm, 0.0)
        log_1m = F.logsigmoid(-z).masked_fill(~cm, 0.0)
        U = log_1m.flip(-1).cumsum(-1).flip(-1) - log_1m
        a = torch.exp(F.logsigmoid(z) + U).masked_fill(~cm, 0.0)
        return a, (1.0 - a.sum(-1)).clamp_min(0.0)
    eff = eff.masked_fill(~cm, NEG_INF)
    if s is None:
        return torch.softmax(eff, dim=-1), None
    lse = torch.logaddexp(torch.logsumexp(eff, -1), s)
    return torch.exp(eff - lse.unsqueeze(-1)), torch.exp(s - lse)


def attn_step(att, h, t, cache, cos, sin, keep=None):
    B, _, D = h.shape
    H, dh = att.n_head, att.d_head
    q, k, v = att.project(h)
    q_raw = q
    if cache.kr is not None:
        cache.kr[:, :, t] = k[:, :, 0]
    if att.rotary is not None:
        q, k = apply_rotary(q, cos[t:t + 1], sin[t:t + 1]), apply_rotary(k, cos[t:t + 1], sin[t:t + 1])
    cache.k[:, :, t], cache.v[:, :, t] = k[:, :, 0], v[:, :, 0]
    cache.alive[:, :, t] = True
    K, V, alive = cache.k[:, :, :t + 1], cache.v[:, :, :t + 1], cache.alive[:, :, :t + 1]
    lg = (up(torch.matmul(q, K.transpose(-1, -2))) * att.scale).squeeze(2)

    dec = None
    if isinstance(att, ForgettingAttention):
        lf = F.logsigmoid(up(att.fgate(h))).view(B, H)
        c_prev = cache.logf[:, :, t - 1] if t > 0 else torch.zeros_like(lf)
        cache.logf[:, :, t] = c_prev + lf
        dec = cache.logf[:, :, t:t + 1] - cache.logf[:, :, :t + 1]
        lg = lg + dec
    lg_read = lg
    if keep is not None and att.rotary is not None:
        M = keep.numel()
        ks = apply_rotary(cache.kr.index_select(2, keep), cos[:M], sin[:M])
        qs = apply_rotary(q_raw, cos[M - 1:M], sin[M - 1:M])
        ls = (up(torch.matmul(qs, ks.transpose(-1, -2))) * att.scale).squeeze(2)
        if dec is not None:
            ls = ls + dec.index_select(-1, keep)
        lg_read = lg.index_copy(-1, keep, ls)

    s = _null_logit(att, h)
    retire = isinstance(att, AbsorbAttention) and att.cfg.retire
    relu_src = retire and att.cfg.retire_src == "relu"
    odds_src = retire and att.cfg.retire_src == "odds"
    selective = isinstance(att, SelectiveAttention)
    R = cache.R[:, :, :t + 1] if (retire or selective) else None
    eff = lg_read if R is None else lg_read - R
    a, n = _read(att, eff, s, alive)
    dead = bool((~alive).any())
    if dead:
        eff_full = eff if lg_read is lg else (lg if R is None else lg - R)
        a_full, _ = _read(att, eff_full, s, torch.ones_like(alive))
        evicted = (a_full * (~alive).to(a_full.dtype)).sum(-1)
    else:
        evicted = torch.zeros(a.shape[:2], device=a.device, dtype=a.dtype)

    if retire or selective:
        bound = isinstance(att, AbsorbAttention) and att.cfg.bind and att.cfg.bind_retire and s is not None
        rho = F.softplus(up(att.rho_proj(h))).view(B, H) if retire and not relu_src else None

        def source(x):
            if relu_src:
                c = F.relu(x)
            elif odds_src:
                c = rho.unsqueeze(-1) * F.relu(x - s.unsqueeze(-1) if bound else x)
            elif retire:
                if bound:
                    c = rho.unsqueeze(-1) * torch.exp(x - torch.logaddexp(torch.logsumexp(x, -1), s).unsqueeze(-1))
                else:
                    c = rho.unsqueeze(-1) * torch.softmax(x, dim=-1)
                if att.cfg.retire_src == "attn_scaled":
                    c = c * (t + 1)
            else:
                c = F.relu(x[:, :1]).expand(B, H, t + 1)
            if retire and att.cfg.retire_share == "mean":
                c = c.mean(1, keepdim=True).expand(B, H, t + 1)
            elif retire and att.cfg.retire_share == "head0":
                c = c[:, :1].expand(B, H, t + 1)
            return c

        contrib = source(lg_read.masked_fill(~alive, NEG_INF))
        if dead:
            contrib = torch.where(alive, contrib, source(lg))
        else:
            contrib = contrib.clone()
        if not att.cfg.retire_bos:
            contrib[..., 0] = 0.0
        contrib[..., t] = 0.0
        cache.R[:, :, :t + 1] += contrib

    o = torch.matmul(a.unsqueeze(2).to(V.dtype), V)
    if isinstance(att, AbsorbAttention) and att.cfg.read_gate:
        g = up(att.gate_x(h)).transpose(1, 2) + torch.einsum("bhnd,hd->bhn", up(o), up(att.gate_o))
        o = o * torch.sigmoid(g).unsqueeze(-1).to(o.dtype)
    if isinstance(att, AbsorbAttention) and att.cfg.out_gate != "none":
        o = o * att.out_gate_values(h).to(o.dtype)
    if isinstance(att, GatedAttention):
        o = o * att.gate_values(h).to(o.dtype)
    return att.merge(o), n, a, evicted


def block_step(blk, x, t, cache, cos, sin, keep=None):
    h = blk.n1(x)
    m, n, a, evicted = attn_step(blk.attn, h, t, cache, cos, sin, keep)
    cfg = blk.cfg
    if cfg.residual == "delta":
        x, _, _ = blk.delta(x, h, m, None if n is None else n.unsqueeze(-1))
    elif cfg.residual == "gain":
        x, _ = blk.gain(x, h, m)
    elif cfg.residual == "ddl":
        x, _ = blk.ddl(x, h, m)
    else:
        x = x + m
    h2 = blk.n2(x)
    f = blk.mlp(h2)
    if cfg.residual == "ddl" and cfg.ddl_mlp:
        x, _ = blk.ddl2(x, h2, f)
    else:
        x = x + f
    return x, a, evicted


def apply_policy(cache, t, a, policy, tau, budget, n_sink, protect0=True):
    if policy == "none":
        return
    alive = cache.alive[:, :, :t + 1]
    if policy == "retire_tau":
        drop = cache.R[:, :, :t + 1] > tau
        if protect0:
            drop[..., 0] = False
        alive &= ~drop
        return
    count = alive.sum(-1)
    over = count > budget
    if not over.any():
        return
    if policy in ("window", "stream"):
        idx = torch.arange(t + 1, device=alive.device)
        score = torch.where(alive & (idx >= n_sink), idx.float(), torch.full_like(idx, 10**9).float())
        victim = score.argmin(-1)
    elif policy == "tova":
        score = a.masked_fill(~alive, float("inf")).clone()
        score[..., t] = float("inf")
        victim = score.argmin(-1)
    elif policy == "tova_layer":
        score = a.mean(1, keepdim=True).masked_fill(~alive[:, :1], float("inf")).clone()
        score[..., t] = float("inf")
        victim = score.argmin(-1).expand(alive.shape[:2])
    elif policy == "retire_budget":
        score = cache.R[:, :, :t + 1].masked_fill(~alive, -1.0).clone()
        if protect0:
            score[..., 0] = -1.0
        score[..., t] = -1.0
        victim = score.argmax(-1)
    else:
        raise ValueError(policy)
    b, h = over.nonzero(as_tuple=True)
    alive[b, h, victim[b, h]] = False


@torch.no_grad()
def decode(model, idx, policy="none", tau=math.inf, budget=None, n_sink=4, targets=None):
    cfg = model.cfg
    if getattr(cfg, "retire_iters", 1) > 1:
        raise NotImplementedError("decode implements retire_iters = 1 only")
    B, N = idx.shape
    dev = idx.device
    dtype = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32
    H, dh = cfg.n_head, cfg.d_model // cfg.n_head
    L = len(model.blocks)
    budgets = list(budget) if isinstance(budget, (list, tuple)) else [budget] * L
    stream = policy == "stream"
    if stream and any(b is None or b < n_sink for b in budgets):
        raise ValueError(f"policy 'stream' needs budget >= n_sink, got {budget} with n_sink={n_sink}")
    caches = [LayerCache(B, H, N, dh, dev, dtype, raw=stream) for _ in model.blocks]
    rot = model.blocks[0].attn.rotary
    cos, sin = rot(N, dev) if rot is not None else (None, None)
    pos = torch.arange(N, device=dev)
    logits = None
    if targets is not None:
        nll = torch.zeros(B, N, device=dev)
        correct = torch.zeros(B, N, device=dev, dtype=torch.bool)
    kv_frac = torch.zeros(L, device=dev)
    kv_peak = torch.zeros(L, device=dev)
    ev_sum = torch.zeros(L, device=dev)
    ev_max = torch.zeros(L, device=dev)
    protect0 = not getattr(cfg, "retire_bos", False)
    for t in range(N):
        x = model.emb(idx[:, t:t + 1])
        for li, (blk, c) in enumerate(zip(model.blocks, caches)):
            b = budgets[li]
            keep = torch.cat([pos[:n_sink], pos[t - b + n_sink:t + 1]]) if stream and t > b else None
            x, a, ev = block_step(blk, x, t, c, cos, sin, keep)
            ev_sum[li] += ev.float().mean()
            ev_max[li] = torch.maximum(ev_max[li], ev.float().max())
            apply_policy(c, t, a, policy, tau, b, n_sink, protect0)
            cnt = c.alive[:, :, :t + 1].sum(-1).float()
            kv_frac[li] += (cnt / (t + 1)).mean()
            kv_peak[li] = torch.maximum(kv_peak[li], cnt.max())
        y = model.head(model.norm(x))[:, 0]
        if targets is None:
            if logits is None:
                logits = y.new_empty(B, N, y.shape[-1])
            logits[:, t] = y
        else:
            tg = targets[:, t]
            nll[:, t] = F.cross_entropy(y.float(), tg, ignore_index=-100, reduction="none")
            correct[:, t] = (y.argmax(-1) == tg) & (tg != -100)
    mem = {"kv_frac_mean": float(kv_frac.mean() / N), "kv_frac_per_layer": (kv_frac / N).tolist(),
           "kv_final_frac": float(torch.stack([c.alive[:, :, :N].float().mean() for c in caches]).mean()),
           "kv_final_per_layer": [float(c.alive[:, :, :N].float().mean()) for c in caches],
           "kv_peak_keys": float(kv_peak.max()), "kv_peak_per_layer": kv_peak.tolist(),
           "evicted_mass_mean": float(ev_sum.mean() / N), "evicted_mass_per_layer": (ev_sum / N).tolist(),
           "evicted_mass_max": float(ev_max.max()), "seq_len": N}
    if targets is not None:
        return {"nll": nll, "correct": correct}, mem
    return logits, mem


@torch.no_grad()
def eval_eviction(model, task, settings, batch=16, n_batches=2, scale=1, device="cuda", amp_dtype=None, seed=4321,
                  batches=None):
    from eval import autocast
    model.eval()
    if batches is None and hasattr(task, "fixed_batches"):
        batches = list(task.fixed_batches(batch, n_batches, scale=scale, device=device))
    elif batches is None:
        gen = torch.Generator(device=device).manual_seed(seed)
        batches = [task.sample(batch, scale=scale, gen=gen, device=device) for _ in range(n_batches)]
    res = []
    for st in settings:
        correct = total = 0
        nll = 0.0
        mems = []
        for d in batches:
            with autocast(torch.device(device), amp_dtype):
                tok, mem = decode(model, d["input"], targets=d["target"], **st)
            correct += int(tok["correct"].sum())
            total += int((d["target"] != -100).sum())
            nll += float(tok["nll"].double().sum())
            mems.append(mem)
        r = dict(st)
        r["acc"] = correct / max(total, 1)
        r["loss"] = nll / max(total, 1)
        r["ppl"] = math.exp(min(r["loss"], 50))
        for k in ("kv_frac_mean", "kv_final_frac", "evicted_mass_mean"):
            r[k] = sum(x[k] for x in mems) / len(mems)
        for k in ("kv_frac_per_layer", "kv_final_per_layer", "evicted_mass_per_layer"):
            r[k] = [sum(v) / len(mems) for v in zip(*(x[k] for x in mems))]
        r["kv_peak_keys"] = max(x["kv_peak_keys"] for x in mems)
        r["kv_peak_per_layer"] = [max(v) for v in zip(*(x["kv_peak_per_layer"] for x in mems))]
        r["evicted_mass_max"] = max(x["evicted_mass_max"] for x in mems)
        r["seq_len"] = mems[0]["seq_len"]
        r["tokens"] = total
        res.append(r)
    model.train()
    return res
