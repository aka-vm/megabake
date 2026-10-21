"""The pinned CUTLASS upstream file next to MegaBake's GEMM fork is unmodified."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cutlass_upstream_copy_is_unmodified():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "vendor_cutlass.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
