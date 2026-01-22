from __future__ import annotations
from dataclasses import dataclass
from .ir import PositionToFreqCalib, ToneId


@dataclass(frozen=True)
class LinearPositionToFreqCalib(PositionToFreqCalib):
    """
    Simplest possible example calibration:
      df = slope * dx

    Later you might make slope per-tone, axis-dependent, or use a matrix from measurements.
    """

    slope_hz_per_um: float

    def df_hz(self, tone_id: ToneId, dx_um: float, axis: str) -> float:
        return self.slope_hz_per_um * float(dx_um)
