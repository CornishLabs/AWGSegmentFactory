"""Calibration helpers for higher-level, position-like operations.

The core compilation pipeline operates on frequency deltas directly. Optical-power
calibrations are applied during segment compilation via
`QIRtoSamplesSegmentCompiler.compile_to_card_int16(...)` or
`QIRtoSamplesSegmentCompiler.compile_to_voltage_mV(...)`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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

    This is applied during sample synthesis in
    `QIRtoSamplesSegmentCompiler.compile_to_card_int16(...)` and
    `QIRtoSamplesSegmentCompiler.compile_to_voltage_mV(...)`, and is also used for
    crest-factor phase optimisation so the optimiser sees the correct
    RF tone weights.

    Units convention used across this package:
    - input optical powers are arbitrary units ("arb"), but must be consistent with
      the calibration data used to fit the model.
    - returned RF amplitudes are in mV.

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
        """Return RF synthesis amplitudes in mV (same shape as `optical_powers`)."""
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
    Using x = normalized(freq_hz, freq_min_hz, freq_max_hz) and polynomials in x:
        g(freq)  = polyval(g_poly, x)
        v0(freq) = sqrt(polyval(v0_a_poly, x)^2 + min_v0_sq)

    Inversion (used to synthesize RF amplitudes from desired optical powers):
        rf_amp = v0(freq) * (2/π) * arcsin(sqrt(optical_power / g(freq)))

    Units:
    - `g(freq)` is optical power in arbitrary units (arb).
    - `v0(freq)` is RF amplitude in mV (it is a function of frequency, not a frequency).
    - `rf_amp` returned by inversion is in mV.

    Notes:
    - This is globally invertible for optical_power in [0, g(freq)] if you restrict to the
      first lobe (rf_amp ∈ [0, v0]). For robustness, we clamp optical_power/g into [0, 1-eps).
    - Designed to support both NumPy (`xp=np`) and CuPy (`xp=cupy`).
    """

    g_poly_high_to_low: Tuple[float, ...]
    v0_a_poly_high_to_low: Tuple[float, ...]
    freq_min_hz: float
    freq_max_hz: float
    traceability_string: str = ""
    min_g: float = 1e-12
    min_v0_sq: float = 1e-12
    y_eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(tuple(self.g_poly_high_to_low)) == 0:
            raise ValueError("g_poly_high_to_low must not be empty")
        if len(tuple(self.v0_a_poly_high_to_low)) == 0:
            raise ValueError("v0_a_poly_high_to_low must not be empty")
        if not np.isfinite(float(self.freq_min_hz)):
            raise ValueError("freq_min_hz must be finite")
        if not np.isfinite(float(self.freq_max_hz)):
            raise ValueError("freq_max_hz must be finite")
        if float(self.freq_max_hz) <= float(self.freq_min_hz):
            raise ValueError("freq_max_hz must be > freq_min_hz")
        if not np.isfinite(float(self.min_g)) or float(self.min_g) <= 0:
            raise ValueError("min_g must be finite and > 0")
        if not np.isfinite(float(self.min_v0_sq)) or float(self.min_v0_sq) <= 0:
            raise ValueError("min_v0_sq must be finite and > 0")
        if not np.isfinite(float(self.y_eps)) or not (0.0 < float(self.y_eps) < 1.0):
            raise ValueError("y_eps must be finite and in (0, 1)")
        object.__setattr__(self, "traceability_string", str(self.traceability_string))

    def _freq_mid_hz(self) -> float:
        return 0.5 * (float(self.freq_min_hz) + float(self.freq_max_hz))

    def _freq_halfspan_hz(self) -> float:
        return 0.5 * (float(self.freq_max_hz) - float(self.freq_min_hz))

    def _x_from_freq(self, freqs_hz: Any, *, xp: Any) -> Any:
        f = xp.asarray(freqs_hz, dtype=float)
        x = (f - self._freq_mid_hz()) / self._freq_halfspan_hz()
        return xp.clip(x, -1.0, 1.0)

    def g_of_freq(self, freqs_hz: Any, *, xp: Any = np) -> Any:
        x = self._x_from_freq(freqs_hz, xp=xp)
        g = _polyval_horner(tuple(self.g_poly_high_to_low), x, xp=xp)
        return xp.maximum(g, float(self.min_g))

    def v0_of_freq_mV(self, freqs_hz: Any, *, xp: Any = np) -> Any:
        x = self._x_from_freq(freqs_hz, xp=xp)
        a0 = _polyval_horner(tuple(self.v0_a_poly_high_to_low), x, xp=xp)
        return xp.sqrt((a0 * a0) + float(self.min_v0_sq))

    def rf_amps(
        self,
        freqs_hz: Any,
        optical_powers: Any,
        *,
        logical_channel: str,
        xp: Any = np,
    ) -> Any:
        p_opt = xp.asarray(optical_powers, dtype=float)
        # Single-channel model; `logical_channel` is accepted for interface consistency.
        g = self.g_of_freq(freqs_hz, xp=xp)
        v0 = self.v0_of_freq_mV(freqs_hz, xp=xp)

        y = xp.maximum(p_opt, 0.0) / g
        y = xp.clip(y, 0.0, 1.0 - float(self.y_eps))
        return v0 * (2.0 / float(np.pi)) * xp.arcsin(xp.sqrt(y))

    @property
    def best_freq_hz(self) -> int:
        ff = np.linspace(
            float(self.freq_min_hz),
            float(self.freq_max_hz),
            num=4096,
            dtype=float,
        )
        gg = np.asarray(self.g_of_freq(ff, xp=np), dtype=float)
        finite = np.isfinite(gg)
        if not np.any(finite):
            return int(round(self._freq_mid_hz()))
        gg_safe = np.where(finite, gg, -np.inf)
        i = int(np.argmax(gg_safe))
        return int(round(float(ff[i])))

    def serialise(self) -> Dict[str, Any]:
        return {
            "g_poly_high_to_low": [float(x) for x in tuple(self.g_poly_high_to_low)],
            "v0_a_poly_high_to_low": [float(x) for x in tuple(self.v0_a_poly_high_to_low)],
            "freq_min_hz": float(self.freq_min_hz),
            "freq_max_hz": float(self.freq_max_hz),
            "traceability_string": str(self.traceability_string),
            "best_freq_hz": int(self.best_freq_hz),
            "min_g": float(self.min_g),
            "min_v0_sq": float(self.min_v0_sq),
            "y_eps": float(self.y_eps),
        }

    @classmethod
    def deserialise(cls, data: Mapping[str, Any]) -> "AODSin2Calib":
        return cls(
            g_poly_high_to_low=tuple(float(x) for x in list(data["g_poly_high_to_low"])),
            v0_a_poly_high_to_low=tuple(float(x) for x in list(data["v0_a_poly_high_to_low"])),
            freq_min_hz=float(data["freq_min_hz"]),
            freq_max_hz=float(data["freq_max_hz"]),
            traceability_string=str(data.get("traceability_string", "")),
            min_g=float(data.get("min_g", 1e-12)),
            min_v0_sq=float(data.get("min_v0_sq", 1e-12)),
            y_eps=float(data.get("y_eps", 1e-6)),
        )


@dataclass(frozen=True)
class AWGPhysicalSetupInfo(OpticalPowerToRFAmpCalib):
    """
    Physical AWG setup information:
    - logical -> hardware channel routing
    - optional per-hardware-channel optical-power calibration.

    If a channel has no calibration (`None`), amplitudes are interpreted as direct RF
    synthesis amplitudes (mV by convention) for that channel.

    If a channel has an `AODSin2Calib`, amplitudes are interpreted as desired optical
    powers (arb), and are converted to RF amplitudes (mV) during synthesis.
    """

    logical_to_hardware_map: Dict[str, int]
    channel_calibrations: Tuple[Optional[AODSin2Calib], ...] = tuple()

    def __post_init__(self) -> None:
        mapping = {
            str(k): int(v) for k, v in dict(self.logical_to_hardware_map).items()
        }
        if not mapping:
            raise ValueError("logical_to_hardware_map must not be empty")

        hw = sorted(set(mapping.values()))
        if any(h < 0 for h in hw):
            raise ValueError(
                f"Hardware channel indices must be >= 0, got {hw}"
            )
        if hw != list(range(len(hw))):
            raise ValueError(
                "Hardware channel indices must be contiguous 0..N-1; "
                f"got {hw}"
            )
        if len(hw) != len(mapping):
            raise ValueError("logical_to_hardware_map must be one-to-one")
        n_ch = len(hw)

        if len(self.channel_calibrations) == 0:
            calibs: Tuple[Optional[AODSin2Calib], ...] = tuple(None for _ in range(n_ch))
        else:
            calibs = tuple(self.channel_calibrations)
            if len(calibs) != n_ch:
                raise ValueError(
                    "channel_calibrations length must match the number of hardware channels; "
                    f"got {len(calibs)} for N_ch={n_ch}"
                )
        for i, calib in enumerate(calibs):
            if calib is not None and not isinstance(calib, AODSin2Calib):
                raise TypeError(
                    "channel_calibrations must contain AODSin2Calib or None; "
                    f"index {i} has {type(calib).__name__}"
                )

        object.__setattr__(self, "logical_to_hardware_map", mapping)
        object.__setattr__(self, "channel_calibrations", calibs)

    @property
    def N_ch(self) -> int:
        return len(self.channel_calibrations)

    def hardware_channel(self, logical_channel: str) -> int:
        key = str(logical_channel)
        if key not in self.logical_to_hardware_map:
            raise KeyError(
                f"Unknown logical_channel {key!r}; available: {sorted(self.logical_to_hardware_map.keys())}"
            )
        return int(self.logical_to_hardware_map[key])

    @classmethod
    def identity(cls, logical_channels: Sequence[str]) -> "AWGPhysicalSetupInfo":
        mapping = {str(lc): i for i, lc in enumerate(tuple(logical_channels))}
        return cls(logical_to_hardware_map=mapping)

    def rf_amps(
        self,
        freqs_hz: Any,
        optical_powers: Any,
        *,
        logical_channel: str,
        xp: Any = np,
    ) -> Any:
        hw_idx = self.hardware_channel(logical_channel)
        calib = self.channel_calibrations[hw_idx]
        if calib is None:
            return xp.asarray(optical_powers, dtype=float)
        return calib.rf_amps(
            freqs_hz,
            optical_powers,
            logical_channel=str(logical_channel),
            xp=xp,
        )

    def serialise(self) -> Dict[str, Any]:
        return {
            "logical_to_hardware_map": {
                str(k): int(v) for k, v in self.logical_to_hardware_map.items()
            },
            "channel_calibrations": [
                None if c is None else c.serialise() for c in self.channel_calibrations
            ],
        }

    @classmethod
    def deserialise(cls, data: Mapping[str, Any]) -> "AWGPhysicalSetupInfo":
        raw = data.get("channel_calibrations", [])
        if not isinstance(raw, list):
            raise ValueError("AWGPhysicalSetupInfo.channel_calibrations must be a list")
        calibs: list[Optional[AODSin2Calib]] = []
        for item in raw:
            if item is None:
                calibs.append(None)
            elif isinstance(item, Mapping):
                calibs.append(AODSin2Calib.deserialise(item))
            else:
                raise ValueError(
                    "AWGPhysicalSetupInfo.channel_calibrations entries must be object or null"
                )
        return cls(
            logical_to_hardware_map={
                str(k): int(v)
                for k, v in dict(data.get("logical_to_hardware_map", {})).items()
            },
            channel_calibrations=tuple(calibs),
        )

    def to_file(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.serialise(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AWGPhysicalSetupInfo":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("AWG physical setup file must contain a JSON object")
        return cls.deserialise(payload)
