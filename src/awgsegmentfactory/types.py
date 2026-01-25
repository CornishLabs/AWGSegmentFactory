"""Shared type aliases used across the compilation pipeline."""

from __future__ import annotations

from typing import TypeAlias

# Mapping from user-defined logical channel names (e.g. "H", "V") to hardware channel indices.
ChannelMap: TypeAlias = dict[str, int]

