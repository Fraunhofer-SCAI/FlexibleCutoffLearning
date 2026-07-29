from __future__ import annotations

from typing import Any


def data_keys(data: Any) -> list[str]:
    if hasattr(data, "keys"):
        keys = data.keys
        if isinstance(keys, list):
            return list(keys)
        if callable(keys):
            return list(keys())
    if isinstance(data, dict):
        return list(data.keys())
    raise TypeError(f"Could not determine keys for object of type {type(data)}")
