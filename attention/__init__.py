from .absorb import AbsorbAttention
from .baseline import VanillaAttention
from .variants import (
    ForgettingAttention,
    GatedAttention,
    SelectiveAttention,
    SinkAttention,
    StickBreakingAttention,
)

ATTENTION = {
    "vanilla": VanillaAttention,
    "gated": GatedAttention,
    "sink": SinkAttention,
    "selective": SelectiveAttention,
    "fox": ForgettingAttention,
    "stickbreak": StickBreakingAttention,
    "absorb": AbsorbAttention,
}


def make_attention(cfg):
    return ATTENTION[cfg.attn](cfg)
