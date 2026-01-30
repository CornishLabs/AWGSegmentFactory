"""Phase optimisation helpers for multitone waveforms.

This module is aimed at AWG/AOD use-cases where you drive a multitone RF waveform:

    y(t) = Σ_k a_k sin(2π f_k t + φ_k)

The per-tone phases φ_k can be chosen to reduce the waveform crest factor
(`max(|y|) / rms(y)`), which helps avoid amplifier saturation and reduces
intermodulation artefacts.

Notes
-----
- Crest factor is computed over a *discrete* time grid. For AWG usage, pass a
  representative `t_s` grid (e.g. sample times) or `(sample_rate_hz, n_samples)`.
- This code optimises phases only; spot intensities are primarily controlled by
  tone amplitudes (subject to AOD/amplifier nonidealities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt


_TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class MultisineBasis:
    """Precomputed sin/cos basis for fast phase-only evaluations."""

    t_s: npt.NDArray[np.floating]
    freqs_hz: npt.NDArray[np.floating]
    amps: npt.NDArray[np.floating]
    sin_wt: npt.NDArray[np.floating]  # (n_time, n_tones)
    cos_wt: npt.NDArray[np.floating]  # (n_time, n_tones)

    @property
    def n_tones(self) -> int:
        return int(self.freqs_hz.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.t_s.shape[0])

    @classmethod
    def from_tones(
        cls,
        *,
        t_s: npt.ArrayLike,
        freqs_hz: Sequence[float],
        amps: Sequence[float],
    ) -> "MultisineBasis":
        t = np.asarray(t_s, dtype=float).reshape(-1)
        f = np.asarray(freqs_hz, dtype=float).reshape(-1)
        a = np.asarray(amps, dtype=float).reshape(-1)

        if t.size == 0:
            raise ValueError("t_s must be non-empty")
        if f.size == 0:
            raise ValueError("freqs_hz must be non-empty")
        if f.shape != a.shape:
            raise ValueError("freqs_hz and amps must have the same length")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(a)):
            raise ValueError("t_s, freqs_hz, and amps must be finite")

        wt = (2.0 * np.pi) * t[:, None] * f[None, :]
        return cls(
            t_s=t,
            freqs_hz=f,
            amps=a,
            sin_wt=np.sin(wt),
            cos_wt=np.cos(wt),
        )


def schroeder_phases_rad(n_tones: int) -> npt.NDArray[np.floating]:
    """
    Return a Schroeder/Newman-style phase seed for crest-factor reduction.

    Uses the common quadratic phase prescription:

        φ_k = -π * k * (k - 1) / N   for k = 1..N

    Returned phases are wrapped to [0, 2π) with φ_1 = 0.
    """
    n = int(n_tones)
    if n <= 0:
        raise ValueError("n_tones must be > 0")
    k = np.arange(1, n + 1, dtype=float)
    phi = -(np.pi * k * (k - 1.0) / float(n))
    phi = np.mod(phi - phi[0], _TWO_PI)
    return phi


def waveform_from_phases(
    basis: MultisineBasis, phases_rad: Sequence[float]
) -> npt.NDArray[np.floating]:
    """Synthesize the multisine waveform y(t) over the basis time grid."""
    phases = np.asarray(phases_rad, dtype=float).reshape(-1)
    if phases.shape != (basis.n_tones,):
        raise ValueError(
            f"phases_rad must have shape ({basis.n_tones},), got {phases.shape}"
        )
    a_cos = basis.amps * np.cos(phases)
    a_sin = basis.amps * np.sin(phases)
    return (basis.sin_wt @ a_cos) + (basis.cos_wt @ a_sin)


def crest_factor(y: npt.ArrayLike) -> float:
    """Return crest factor = max(|y|) / rms(y)."""
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size == 0:
        raise ValueError("y must be non-empty")
    rms = float(np.sqrt(np.mean(yy * yy)))
    if rms == 0.0:
        return float("inf")
    peak = float(np.max(np.abs(yy)))
    return peak / rms


def minimise_crest_factor_phases(
    freqs_hz: Sequence[float],
    amps: Sequence[float],
    *,
    t_s: Optional[npt.ArrayLike] = None,
    sample_rate_hz: Optional[float] = None,
    n_samples: int = 4096,
    passes: int = 1,
    xatol_rad: float = 1e-3,
    method: Literal["schroeder", "coordinate"] = "coordinate",
    output: Literal["rad", "deg"] = "rad",
    phases_init_rad: Optional[Sequence[float]] = None,
    fixed_mask: Optional[Sequence[bool]] = None,
) -> npt.NDArray[np.floating]:
    """
    Choose phases that reduce crest factor for a multitone waveform.

    Parameters
    ----------
    freqs_hz, amps:
        Per-tone frequencies and amplitudes.
    t_s:
        Optional explicit time grid for evaluating crest factor.
    sample_rate_hz, n_samples:
        If `t_s` is not provided, use `t_s = arange(n_samples) / sample_rate_hz`.
    passes:
        Number of coordinate-descent passes over tones when `method="coordinate"`.
    xatol_rad:
        Absolute tolerance used by the 1D bounded search per tone.
    method:
        - "schroeder": return the analytical seed phases only.
        - "coordinate": coordinate-descent refinement (optimise each phase in turn).
    output:
        Return phases in radians ("rad") or degrees ("deg").
    phases_init_rad:
        Optional initial phases (radians). If omitted, uses a Schroeder-style seed.
    fixed_mask:
        Optional boolean mask of tones to keep fixed (True = fixed). Useful for
        "carry some phases, optimise the rest" workflows.
    """
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    a = np.asarray(amps, dtype=float).reshape(-1)
    if f.size == 0:
        raise ValueError("freqs_hz must be non-empty")
    if f.shape != a.shape:
        raise ValueError("freqs_hz and amps must have the same length")
    n_tones = int(f.shape[0])

    if t_s is None:
        if sample_rate_hz is None:
            raise ValueError("Provide either t_s=... or sample_rate_hz=...")
        fs = float(sample_rate_hz)
        if fs <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        n = int(n_samples)
        if n <= 0:
            raise ValueError("n_samples must be > 0")
        t_s = np.arange(n, dtype=float) / fs

    if phases_init_rad is None:
        phases = schroeder_phases_rad(n_tones).astype(float, copy=True)
    else:
        phases = np.asarray(phases_init_rad, dtype=float).reshape(-1).copy()
        if phases.shape != (n_tones,):
            raise ValueError(
                f"phases_init_rad must have shape ({n_tones},), got {phases.shape}"
            )

    if fixed_mask is None:
        fixed = np.zeros((n_tones,), dtype=bool)
    else:
        fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed.shape != (n_tones,):
            raise ValueError(
                f"fixed_mask must have shape ({n_tones},), got {fixed.shape}"
            )

    if (
        method == "schroeder"
        or passes <= 0
        or n_tones == 1
        or np.all(fixed)
        or np.all(a == 0.0)
    ):
        out = phases
        return np.rad2deg(out) if output == "deg" else out
    if method != "coordinate":
        raise ValueError("method must be 'schroeder' or 'coordinate'")

    basis = MultisineBasis.from_tones(t_s=t_s, freqs_hz=f, amps=a)
    try:
        from scipy.optimize import minimize_scalar  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for coordinate phase refinement") from exc

    y = waveform_from_phases(basis, phases)

    def _crest_fast(yy: npt.NDArray[np.floating]) -> float:
        rms = float(np.sqrt(np.mean(yy * yy)))
        if rms == 0.0:
            return float("inf")
        return float(np.max(np.abs(yy))) / rms

    for _ in range(int(passes)):
        for k in range(basis.n_tones):
            if bool(fixed[k]):
                continue
            sin_col = basis.sin_wt[:, k]
            cos_col = basis.cos_wt[:, k]
            amp_k = float(basis.amps[k])
            if amp_k == 0.0:
                continue

            phi0 = float(phases[k])
            contrib0 = amp_k * (sin_col * np.cos(phi0) + cos_col * np.sin(phi0))
            y_without = y - contrib0

            def obj(phi: float) -> float:
                contrib = amp_k * (sin_col * np.cos(phi) + cos_col * np.sin(phi))
                return _crest_fast(y_without + contrib)

            res = minimize_scalar(
                obj,
                bounds=(0.0, _TWO_PI),
                method="bounded",
                options={"xatol": float(xatol_rad)},
            )
            phi1 = float(res.x) % _TWO_PI
            phases[k] = phi1
            y = y_without + amp_k * (sin_col * np.cos(phi1) + cos_col * np.sin(phi1))

    out = phases
    return np.rad2deg(out) if output == "deg" else out
