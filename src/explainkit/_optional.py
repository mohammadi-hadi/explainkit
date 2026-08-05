"""Import optional backends with an error message that says how to fix it."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, *, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is required for this feature; "
            f'install it with:  pip install "explainkit[{extra}]"'
        ) from exc
