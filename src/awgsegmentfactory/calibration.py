"""Calibration helpers for higher-level, position-like operations.

Calibrations are currently stored on `IntentIR.calibrations` for future integration
into builder ops. The core compilation pipeline operates on frequency deltas directly.
"""

from __future__ import annotations
from dataclasses import dataclass
from .intent_ir import PositionToFreqCalib, ToneId


@dataclass(frozen=True)
class LinearPositionToFreqCalib(PositionToFreqCalib):
    """
    Simplest possible example calibration:
      df = slope * dx

    Later you might make slope per-tone, axis-dependent, or use a matrix from measurements.
    """

    slope_hz_per_um: float

    def df_hz(self, tone_id: ToneId, dx_um: float, logical_channel: str) -> float:
        """Return `slope_hz_per_um * dx_um` (ignores tone/channel)."""
        return self.slope_hz_per_um * float(dx_um)


### Designing calibrations

class AODCalib():
    """Placeholder for future AOD calibration models."""

    pass
