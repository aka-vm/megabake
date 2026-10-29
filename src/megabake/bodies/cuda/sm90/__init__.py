"""Hopper (sm90a): TMA/WGMMA GEMMs and softmax attention."""

from ...interface import Generation
from .attention import AttentionBodyProvider
from .gemm import G0RangeBodyProvider, launch_options

GENERATION = Generation(
    architectures=frozenset({"sm90a"}),
    providers=(G0RangeBodyProvider(), AttentionBodyProvider()),
    launch_options=launch_options,
)
