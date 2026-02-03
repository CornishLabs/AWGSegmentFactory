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
class AODTanh2Calib(OpticalPowerToRFAmpCalib):
    """
    First-order diffraction-efficiency calibration with smooth saturation.

    Model (per logical channel):
        optical_power ≈ g(freq) * tanh^2(rf_amp / v0(freq))

    Using x = (freq_hz - mid_hz) / halfspan_hz and polynomials in x:
        g(freq)  = polyval(g_poly, x)
        v0(freq) = sqrt(polyval(v0_a_poly, x)^2 + min_v0_sq)

    Inversion (used to synthesize RF amplitudes from desired optical powers):
        rf_amp = amp_scale * v0(freq) * arctanh(sqrt(optical_power / g(freq)))

    Notes:
    - This is globally invertible for optical_power in [0, g(freq)), and is monotone in rf_amp >= 0.
    - For numerical stability we clamp the ratio optical_power/g into [0, 1-eps).
    - Designed to support both NumPy (`xp=np`) and CuPy (`xp=cupy`).
    """

    g_poly_by_logical_channel: Dict[str, Tuple[float, ...]]
    v0_a_poly_by_logical_channel: Dict[str, Tuple[float, ...]]
    freq_mid_hz: float
    freq_halfspan_hz: float
    amp_scale: float = 1.0
    min_g: float = 1e-12
    min_v0_sq: float = 1e-12
    y_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.freq_mid_hz)):
            raise ValueError("freq_mid_hz must be finite")
        if not np.isfinite(float(self.freq_halfspan_hz)) or float(self.freq_halfspan_hz) <= 0:
            raise ValueError("freq_halfspan_hz must be finite and > 0")
        if not np.isfinite(float(self.amp_scale)) or float(self.amp_scale) <= 0:
            raise ValueError("amp_scale must be finite and > 0")
        if not np.isfinite(float(self.min_g)) or float(self.min_g) <= 0:
            raise ValueError("min_g must be finite and > 0")
        if not np.isfinite(float(self.min_v0_sq)) or float(self.min_v0_sq) <= 0:
            raise ValueError("min_v0_sq must be finite and > 0")
        if not np.isfinite(float(self.y_eps)) or not (0.0 < float(self.y_eps) < 1.0):
            raise ValueError("y_eps must be finite and in (0, 1)")

    def _coeffs(
        self, coeffs_by_logical_channel: Dict[str, Tuple[float, ...]], logical_channel: str
    ) -> Tuple[float, ...]:
        if logical_channel in coeffs_by_logical_channel:
            return coeffs_by_logical_channel[logical_channel]
        if "*" in coeffs_by_logical_channel:
            return coeffs_by_logical_channel["*"]
        if len(coeffs_by_logical_channel) == 1:
            return next(iter(coeffs_by_logical_channel.values()))
        raise KeyError(
            "AODTanh2Calib: no coefficients for logical_channel "
            f"{logical_channel!r} (available: {sorted(coeffs_by_logical_channel.keys())})"
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

        x = (f - float(self.freq_mid_hz)) / float(self.freq_halfspan_hz)
        # Avoid runaway polynomial extrapolation if users pass slightly out-of-range freqs.
        x = xp.clip(x, -1.0, 1.0)

        g = _polyval_horner(self._coeffs(self.g_poly_by_logical_channel, str(logical_channel)), x, xp=xp)
        g = xp.maximum(g, float(self.min_g))

        a0 = _polyval_horner(
            self._coeffs(self.v0_a_poly_by_logical_channel, str(logical_channel)),
            x,
            xp=xp,
        )
        v0 = xp.sqrt((a0 * a0) + float(self.min_v0_sq))

        y = xp.maximum(p_opt, 0.0) / g
        y = xp.clip(y, 0.0, 1.0 - float(self.y_eps))
        rf_amp = v0 * xp.arctanh(xp.sqrt(y))
        return float(self.amp_scale) * rf_amp


class AODCalib:
    """Placeholder for future AOD calibration models (deprecated)."""

    pass
