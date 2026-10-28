from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megabake.platform import HAS_CUDA, HAS_CUTLASS


def pytest_ignore_collect(collection_path: Path, *args, **kwargs) -> bool | None:
    path_str = str(collection_path)
    if "/tests/gpu" in path_str and (
        not HAS_CUDA or not HAS_CUTLASS or not torch.cuda.is_available()
    ):
        return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path_str = str(item.fspath)
        if "/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.cpu)
        elif "/integration/" in path_str:
            item.add_marker(pytest.mark.integration)
            if "sm90" in item.name.lower() or "sm90" in path_str.lower():
                item.add_marker(pytest.mark.sm90)
            else:
                item.add_marker(pytest.mark.cpu)
        elif "/gpu/" in path_str:
            item.add_marker(pytest.mark.gpu)
            if "blackwell" in path_str.lower() or "blackwell" in item.name.lower():
                item.add_marker(pytest.mark.blackwell)
            elif "sm90" in path_str.lower() or "hopper" in path_str.lower():
                item.add_marker(pytest.mark.sm90)
