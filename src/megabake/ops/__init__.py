"""MegaBake's semantic operation families: what each fused region computes.

Each module owns one family: its typed descriptor, which device bodies read
instead of re-deriving the computation from the graph, and how regions
recognize it in the lowered graph. A family a model declares as a PyTorch op
(attention) also defines that op, its eager reference, and its program.

A computation earns a family when a generic body cannot make it fast and its
variants fit in a descriptor's fields; everything else stays per-op regions
and elementwise chains.
"""

from .attention import (
    ATTENTION_OP,
    HF_EAGER_NUMERICS,
    KV_ATTENTION_OP,
    Attention,
    CacheEffects,
    HeadGrouping,
    Mask,
    NumericalPolicy,
    Prologue,
)
from .elementwise import ScalarChain
from .gemm import GemmEpilogue, ViewPlusGemm
from .norm import RMSNorm
from .rope import Rope

# The descriptor a region or fusion candidate carries, if any.
Semantics = Attention | GemmEpilogue | RMSNorm | Rope | ScalarChain | ViewPlusGemm

__all__ = [
    "ATTENTION_OP",
    "HF_EAGER_NUMERICS",
    "KV_ATTENTION_OP",
    "Attention",
    "CacheEffects",
    "GemmEpilogue",
    "HeadGrouping",
    "Mask",
    "NumericalPolicy",
    "Prologue",
    "RMSNorm",
    "Rope",
    "ScalarChain",
    "Semantics",
    "ViewPlusGemm",
]
