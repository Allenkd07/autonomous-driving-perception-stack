"""Name -> implementation registry, so configs can pick implementations by name.

@register("tracker", "bytetrack")
class ByteTrackTracker: ...

tracker = build("tracker", {"type": "bytetrack", "track_buffer": 30})
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")

KINDS = ("detector", "tracker", "lanes", "depth", "distance", "policy", "backend", "source", "sink")

# Packages whose __init__ imports their implementations (and so registers them).
_IMPLEMENTATION_PACKAGES = (
    "perception.backends",
    "perception.detection",
    "perception.tracking",
    "perception.lanes",
    "perception.depth",
    "perception.fusion",
    "perception.planning",
    "perception.io",
)

_registry: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}
_loaded = False


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    if kind not in _registry:
        raise ValueError(f"Unknown kind {kind!r}. Expected one of {KINDS}")

    def decorator(cls: type[T]) -> type[T]:
        existing = _registry[kind].get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"{kind} {name!r} already registered by {existing.__qualname__}")
        _registry[kind][name] = cls
        return cls

    return decorator


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        for package in _IMPLEMENTATION_PACKAGES:
            importlib.import_module(package)
        _loaded = True


def available(kind: str) -> list[str]:
    _ensure_loaded()
    return sorted(_registry[kind])


def get(kind: str, name: str) -> type:
    _ensure_loaded()
    try:
        return _registry[kind][name]
    except KeyError:
        raise KeyError(f"No {kind} named {name!r}. Available: {available(kind)}") from None


def build(kind: str, cfg: Mapping[str, Any]) -> Any:
    """Instantiate from a config mapping: `type` picks the class, the rest are kwargs."""
    params = dict(cfg)
    try:
        name = params.pop("type")
    except KeyError:
        raise ValueError(f"{kind} config is missing 'type': {dict(cfg)}") from None
    return get(kind, name)(**params)
