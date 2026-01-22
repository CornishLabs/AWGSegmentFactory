from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "interactive_grid_debug",
    "LinearFreqToPos",
    "sequence_samples_debug",
]

if TYPE_CHECKING:  # pragma: no cover
    from .plot import LinearFreqToPos, interactive_grid_debug
    from .samples import sequence_samples_debug


def __getattr__(name: str):
    if name in {"interactive_grid_debug", "LinearFreqToPos"}:
        from .plot import LinearFreqToPos, interactive_grid_debug

        return interactive_grid_debug if name == "interactive_grid_debug" else LinearFreqToPos
    if name == "sequence_samples_debug":
        from .samples import sequence_samples_debug

        return sequence_samples_debug
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

