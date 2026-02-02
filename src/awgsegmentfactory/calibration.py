"""Calibration helpers for higher-level, position-like operations.

Calibrations are currently stored on `IntentIR.calibrations` for future integration
into builder ops. The core compilation pipeline operates on frequency deltas directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

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

class OpticalPowerToRFAmpCalib(ABC):
    """
    Calibration interface for turning a *desired optical power* into an RF *synthesis amplitude*.

    Use-case: you want `amps` in the IR to represent optical power (or a proxy for it),
    but the actual multisine waveform that drives the AOD should use RF amplitudes
    derived from a per-tone model (diffraction efficiency, amplifier response, etc).

    This is applied during sample synthesis (`compile_sequence_program(...)`) and is
    also used for crest-factor phase optimisation so the optimiser sees the correct
    RF tone weights.

    Notes:
    - This is "first order": each tone is calibrated independently (no tone interactions).
    - Implementations should be vectorized and support both NumPy (`xp=np`) and (optionally)
      CuPy (`xp=cupy`) for `gpu=True` compilation.
    """

    @abstractmethod
    def rf_amps(
        self,
        freqs_hz: Any,
        optical_powers: Any,
        *,
        logical_channel: str,
        xp: Any = np,
    ) -> Any:
        """Return RF synthesis amplitudes (same shape as `optical_powers`)."""
        raise NotImplementedError


def _polyval_horner(coeffs_high_to_low: Tuple[float, ...], x: Any, *, xp: Any) -> Any:
    """Evaluate a polynomial with coefficients ordered like `numpy.polyval` (high->low)."""
    if len(coeffs_high_to_low) == 0:
        return xp.zeros_like(x, dtype=float)
    y = xp.zeros_like(x, dtype=float) + float(coeffs_high_to_low[0])
    for c in coeffs_high_to_low[1:]:
        y = y * x + float(c)
    return y


@dataclass(frozen=True)
class AODDECalib(OpticalPowerToRFAmpCalib):
    """
    First-order diffraction-efficiency calibration.

    Model (per logical channel):
        optical_power ≈ DE(freq) * rf_power

    With a fixed load impedance, rf_power ∝ rf_amp^2, so we use:
        rf_amp = amp_scale * sqrt(optical_power / DE(freq))

    `DE(freq)` is provided as a polynomial in `x = freq_hz / freq_scale_hz` with coefficients
    ordered high->low (same convention as `numpy.polyval`).
    """

    de_poly_by_logical_channel: Dict[str, Tuple[float, ...]]
    freq_scale_hz: float = 1e6  # evaluate polynomial in "MHz" by default
    amp_scale: float = 1.0
    min_de: float = 1e-12

    def _coeffs(self, logical_channel: str) -> Tuple[float, ...]:
        if logical_channel in self.de_poly_by_logical_channel:
            return self.de_poly_by_logical_channel[logical_channel]
        if "*" in self.de_poly_by_logical_channel:
            return self.de_poly_by_logical_channel["*"]
        if len(self.de_poly_by_logical_channel) == 1:
            return next(iter(self.de_poly_by_logical_channel.values()))
        raise KeyError(
            f"AODDECalib: no DE polynomial for logical_channel {logical_channel!r} "
            f"(available: {sorted(self.de_poly_by_logical_channel.keys())})"
        )

    def rf_amps(
        self,
        freqs_hz: Any,
        optical_powers: Any,
        *,
        logical_channel: str,
        xp: Any = np,
    ) -> Any:
        f = xp.asarray(freqs_hz, dtype=float)
        p_opt = xp.asarray(optical_powers, dtype=float)
        x = f / float(self.freq_scale_hz)
        de = _polyval_horner(self._coeffs(str(logical_channel)), x, xp=xp)
        de = xp.maximum(de, float(self.min_de))
        rf_power = xp.maximum(p_opt, 0.0) / de
        return float(self.amp_scale) * xp.sqrt(rf_power)


class AODCalib:
    """Placeholder for future AOD calibration models (deprecated by `AODDECalib`)."""

    pass
