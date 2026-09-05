"""Metatron: learnable Flower-of-Life geometric modular networks (pure NumPy).

Public API (stable, used by the daycare trainer/evaluator/bridge):

    from metatron import MetatronV2, SCALES, Config
    from metatron import train, verify_capabilities, make_batches, DEFAULT_TEXT

Names are loaded lazily so ``python -m metatron.metatron_v2`` does not import
the module twice.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .metatron_v2 import (  # noqa: F401
        Config, MetatronV2, DEFAULT_TEXT, gelu, layer_norm, make_batches,
        silu, softmax, train, verify_capabilities, SCALES,
    )

__all__ = [
    "Config", "SCALES", "MetatronV2", "train", "verify_capabilities",
    "make_batches", "DEFAULT_TEXT", "gelu", "silu", "softmax", "layer_norm",
]


def __getattr__(name):
    if name in __all__:
        from . import metatron_v2 as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
