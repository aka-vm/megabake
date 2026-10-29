"""Show how MegaBake's CUTLASS GEMM fork differs from its pinned upstream file.

``src/megabake/bodies/cuda/sm90/cutlass_gemm.py`` is the kernel MegaBake runs and edits
directly. ``src/megabake/bodies/cuda/sm90/cutlass_upstream/`` keeps the upstream file
unmodified, with its revision and SHA-256 in ``UPSTREAM``.

    .venv/bin/python tools/vendor_cutlass.py          # diff upstream -> fork
    .venv/bin/python tools/vendor_cutlass.py --stat   # changed line counts only
    .venv/bin/python tools/vendor_cutlass.py --check  # verify the upstream copy's hash

Upgrading CUTLASS: replace the upstream file and ``UPSTREAM``, then merge the
upstream changes into the fork by hand, using this diff as the guide.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODIES = ROOT / "src" / "megabake" / "bodies" / "cuda" / "sm90"
UPSTREAM = BODIES / "cutlass_upstream"
# Upstream file -> MegaBake's fork of it.
FORKS = {"dense_gemm_persistent.py": BODIES / "cutlass_gemm.py"}


def check_upstream() -> None:
    """Fail unless every upstream copy has the SHA-256 recorded in UPSTREAM."""
    for line in (UPSTREAM / "UPSTREAM").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() != "sha256":
            continue
        name, expected = value.split()
        actual = hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"cutlass_upstream/{name} was modified ({actual} != {expected})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only verify the upstream copy")
    parser.add_argument("--stat", action="store_true", help="show changed line counts only")
    args = parser.parse_args()
    check_upstream()
    if args.check:
        return 0
    for name, fork in FORKS.items():
        subprocess.run(
            [
                "git", "diff", "--no-index", *(["--stat"] if args.stat else []),
                str((UPSTREAM / name).relative_to(ROOT)), str(fork.relative_to(ROOT)),
            ],
            cwd=ROOT,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
