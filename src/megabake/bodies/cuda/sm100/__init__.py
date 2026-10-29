"""Blackwell (SM10x/SM12x): a warp-MMA GEMM baseline."""

from ...interface import Generation
from .gemm import BLACKWELL_ARCHITECTURES, BlackwellGemmProvider

GENERATION = Generation(
    architectures=BLACKWELL_ARCHITECTURES,
    providers=(BlackwellGemmProvider(),),
)
