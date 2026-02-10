"""Calibration helpers for higher-level, position-like operations.

The core compilation pipeline operates on frequency deltas directly. Optical-power
calibrations are applied at compile time via `compile_sequence_program(..., optical_power_calib=...)`.
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
class AODSin2Calib(OpticalPowerToRFAmpCalib):
    """
    First-order diffraction-efficiency calibration using the vendor-style sin² response.

    Vendor note (for input power P_in below a saturation/π-power P_sat):
        DE ≈ sin^2( (π/2) * sqrt(P_in / P_sat) )

    If RF drive power is proportional to voltage² (typical into a fixed impedance),
    and your measured axis is RF voltage amplitude, then sqrt(P_in/P_sat) becomes
    `rf_amp / rf_amp_sat`. This yields the model:

        optical_power ≈ g(freq) * sin^2( (π/2) * rf_amp / v0(freq) )

    Single-channel form:
    Using x = (freq_hz - mid_hz) / halfspan_hz and polynomials in x:
        g(freq)  = polyval(g_poly, x)
        v0(freq) = sqrt(polyval(v0_a_poly, x)^2 + min_v0_sq)

    Inversion (used to synthesize RF amplitudes from desired optical powers):
        rf_amp = amp_scale * v0(freq) * (2/π) * arcsin(sqrt(optical_power / g(freq)))

    Notes:
    - This is globally invertible for optical_power in [0, g(freq)] if you restrict to the
      first lobe (rf_amp ∈ [0, v0]). For robustness, we clamp optical_power/g into [0, 1-eps).
    - Designed to support both NumPy (`xp=np`) and CuPy (`xp=cupy`).
    """

    g_poly_high_to_low: Tuple[float, ...]
    v0_a_poly_high_to_low: Tuple[float, ...]
    freq_mid_hz: float
    freq_halfspan_hz: float
    amp_scale: float = 1.0
    min_g: float = 1e-12
    min_v0_sq: float = 1e-12
    y_eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(tuple(self.g_poly_high_to_low)) == 0:
            raise ValueError("g_poly_high_to_low must not be empty")
        if len(tuple(self.v0_a_poly_high_to_low)) == 0:
            raise ValueError("v0_a_poly_high_to_low must not be empty")
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
        x = xp.clip(x, -1.0, 1.0)

        # Single-channel model; `logical_channel` is ignored here and handled by wrappers.
        g = _polyval_horner(tuple(self.g_poly_high_to_low), x, xp=xp)
        g = xp.maximum(g, float(self.min_g))

        a0 = _polyval_horner(tuple(self.v0_a_poly_high_to_low), x, xp=xp)
        v0 = xp.sqrt((a0 * a0) + float(self.min_v0_sq))

        y = xp.maximum(p_opt, 0.0) / g
        y = xp.clip(y, 0.0, 1.0 - float(self.y_eps))
        rf_amp = v0 * (2.0 / float(np.pi)) * xp.arcsin(xp.sqrt(y))
        return float(self.amp_scale) * rf_amp


@dataclass(frozen=True)
class MultiChannelAODSin2Calib(OpticalPowerToRFAmpCalib):
    """
    Multi-channel calibration wrapper using explicit logical->channel-index mapping.

    - `channel_calibs[i]` corresponds to physical channel index `i`.
    - `logical_to_channel_index` maps logical names (e.g. "H", "V") to those indices.
    """

    channel_calibs: Tuple[AODSin2Calib, ...]
    logical_to_channel_index: Dict[str, int]

    def __post_init__(self) -> None:
        calibs = tuple(self.channel_calibs)
        mapping = {str(k): int(v) for k, v in dict(self.logical_to_channel_index).items()}
        if len(calibs) == 0:
            raise ValueError("channel_calibs must contain at least one channel")
        if not mapping:
            raise ValueError("logical_to_channel_index must not be empty")
        n = len(calibs)
        for logical, idx in mapping.items():
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"logical_to_channel_index[{logical!r}]={idx} out of range for {n} channel(s)"
                )
        object.__setattr__(self, "channel_calibs", calibs)
        object.__setattr__(self, "logical_to_channel_index", mapping)

    def channel_index_for_logical_channel(self, logical_channel: str) -> int:
        key = str(logical_channel)
        if key not in self.logical_to_channel_index:
            raise KeyError(
                f"Unknown logical_channel {key!r}; available: {sorted(self.logical_to_channel_index.keys())}"
            )
        return int(self.logical_to_channel_index[key])

    def rf_amps(
        self,
        freqs_hz: Any,
        optical_powers: Any,
        *,
        logical_channel: str,
        xp: Any = np,
    ) -> Any:
        idx = self.channel_index_for_logical_channel(str(logical_channel))
        return self.channel_calibs[int(idx)].rf_amps(
            freqs_hz,
            optical_powers,
            logical_channel=str(logical_channel),
            xp=xp,
        )
