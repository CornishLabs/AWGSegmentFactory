"""Fitting utilities for first-order optical-power calibrations.

The core compiler uses `OpticalPowerToRFAmpCalib` implementations (see `calibration.py`)
to convert per-tone "optical power" (or a proxy for it) into RF synthesis amplitudes.

This module provides reusable tooling for *building those calibrations* from measured
data of the form:
  (rf_freq_hz, rf_amp_mV) -> optical_power

The main supported model is a smooth, globally-invertible saturation curve:
  optical_power(freq, a) ≈ g(freq) * tanh^2(a / v0(freq))

with `g(freq)` and `v0(freq)` each represented as low-degree polynomials in a
normalized frequency coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .calibration import AODTanh2Calib


@dataclass(frozen=True)
class OpticalPowerCalCurve:
    """One measured curve at a fixed RF frequency."""

    freq_hz: float
    rf_amps_mV: np.ndarray  # (n,)
    optical_powers: np.ndarray  # (n,)
    weights: np.ndarray | None = None  # (n,) optional SSE weights

    def cleaned(
        self,
        *,
        clamp_power_nonnegative: bool = True,
        drop_negative_amps: bool = True,
        dedup_amps: bool = True,
    ) -> "OpticalPowerCalCurve":
        a = np.asarray(self.rf_amps_mV, dtype=float).reshape(-1)
        p = np.asarray(self.optical_powers, dtype=float).reshape(-1)
        if a.shape != p.shape:
            raise ValueError("rf_amps_mV and optical_powers must have the same shape")

        w: np.ndarray | None
        if self.weights is None:
            w = None
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if w.shape != a.shape:
                raise ValueError("weights must have the same shape as rf_amps_mV")

        m = np.isfinite(a) & np.isfinite(p)
        if w is not None:
            m = m & np.isfinite(w) & (w >= 0.0)
        if drop_negative_amps:
            m = m & (a >= 0.0)

        a = a[m]
        p = p[m]
        if w is not None:
            w = w[m]

        if clamp_power_nonnegative:
            p = np.maximum(p, 0.0)

        if a.size == 0:
            return OpticalPowerCalCurve(
                freq_hz=float(self.freq_hz),
                rf_amps_mV=a,
                optical_powers=p,
                weights=w,
            )

        order = np.argsort(a, kind="stable")
        a = a[order]
        p = p[order]
        if w is not None:
            w = w[order]

        if dedup_amps:
            uniq_a, inv = np.unique(a, return_inverse=True)
            if uniq_a.size != a.size:
                if w is None:
                    p_sum = np.zeros_like(uniq_a, dtype=float)
                    counts = np.zeros_like(uniq_a, dtype=int)
                    np.add.at(p_sum, inv, p)
                    np.add.at(counts, inv, 1)
                    p = p_sum / np.maximum(counts, 1)
                    a = uniq_a
                    w = None
                else:
                    # Weighted average per duplicate amplitude.
                    p_wsum = np.zeros_like(uniq_a, dtype=float)
                    w_sum = np.zeros_like(uniq_a, dtype=float)
                    np.add.at(p_wsum, inv, w * p)
                    np.add.at(w_sum, inv, w)
                    p = np.divide(p_wsum, np.maximum(w_sum, 1e-30))
                    a = uniq_a
                    w = w_sum

        return OpticalPowerCalCurve(
            freq_hz=float(self.freq_hz),
            rf_amps_mV=a,
            optical_powers=p,
            weights=w,
        )


def curves_from_point_cloud(
    *,
    freqs_hz: Any,
    rf_amps_mV: Any,
    optical_powers: Any,
    freq_round_hz: float | None = None,
    clamp_power_nonnegative: bool = True,
) -> tuple[OpticalPowerCalCurve, ...]:
    """
    Convert scattered points into per-frequency curves.

    If `freq_round_hz` is provided, frequencies are rounded to that grid before grouping.
    """
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    a = np.asarray(rf_amps_mV, dtype=float).reshape(-1)
    p = np.asarray(optical_powers, dtype=float).reshape(-1)
    if f.shape != a.shape or f.shape != p.shape:
        raise ValueError("freqs_hz, rf_amps_mV, optical_powers must have the same shape")

    m = np.isfinite(f) & np.isfinite(a) & np.isfinite(p)
    f = f[m]
    a = a[m]
    p = p[m]

    if f.size == 0:
        return tuple()

    if freq_round_hz is not None:
        q = float(freq_round_hz)
        if not np.isfinite(q) or q <= 0:
            raise ValueError("freq_round_hz must be finite and > 0")
        f_key = np.round(f / q) * q
    else:
        f_key = f

    uniq_f, inv = np.unique(f_key, return_inverse=True)
    curves: list[OpticalPowerCalCurve] = []
    for i, ff in enumerate(uniq_f):
        mask = inv == i
        curve = OpticalPowerCalCurve(freq_hz=float(ff), rf_amps_mV=a[mask], optical_powers=p[mask])
        curve = curve.cleaned(clamp_power_nonnegative=clamp_power_nonnegative)
        if curve.rf_amps_mV.size > 0:
            curves.append(curve)
    curves.sort(key=lambda c: float(c.freq_hz))
    return tuple(curves)


def curves_from_de_rf_calibration_dict(de_rf: Mapping[str, Any]) -> tuple[OpticalPowerCalCurve, ...]:
    """
    Adapter for the DE-compensation JSON format used in `examples/814_H_calFile_*.txt`.

    The returned curves treat the stored "Diffraction Efficiency" as `optical_powers`.
    """
    keys = sorted(de_rf.keys(), key=lambda k: float(k))
    curves: list[OpticalPowerCalCurve] = []
    for k in keys:
        entry = de_rf[k]
        freq_hz = float(k) * 1e6  # file stores MHz keys as strings
        a = np.asarray(entry["RF Amplitude (mV)"], dtype=float)
        p = np.asarray(entry["Diffraction Efficiency"], dtype=float)
        curve = OpticalPowerCalCurve(freq_hz=freq_hz, rf_amps_mV=a, optical_powers=p).cleaned()
        if curve.rf_amps_mV.size > 0:
            curves.append(curve)
    return tuple(curves)


@dataclass(frozen=True)
class Tanh2PolyFitResult:
    """Fit result for `optical_power ≈ g(freq) * tanh^2(amp / v0(freq))`."""

    degree_g: int
    degree_v0: int
    freq_mid_hz: float
    freq_halfspan_hz: float
    min_v0_sq_mV2: float
    coeffs_g_high_to_low: Tuple[float, ...]
    coeffs_v0_a_high_to_low: Tuple[float, ...]
    freqs_hz: np.ndarray  # (n_freq,)
    g_fit_by_freq: np.ndarray  # (n_freq,)
    v0_mV_by_freq: np.ndarray  # (n_freq,)
    rmse: float
    max_abs_resid: float
    sse: float
    opt_success: bool
    opt_message: str
    opt_nfev: int

    def x_from_freq_hz(self, freqs_hz: Any) -> np.ndarray:
        f = np.asarray(freqs_hz, dtype=float)
        x = (f - float(self.freq_mid_hz)) / float(self.freq_halfspan_hz)
        return np.clip(x, -1.0, 1.0)

    def g_of_freq(self, freqs_hz: Any) -> np.ndarray:
        x = self.x_from_freq_hz(freqs_hz)
        return np.polyval(np.asarray(self.coeffs_g_high_to_low, dtype=float), x)

    def v0_of_freq_mV(self, freqs_hz: Any) -> np.ndarray:
        x = self.x_from_freq_hz(freqs_hz)
        a0 = np.polyval(np.asarray(self.coeffs_v0_a_high_to_low, dtype=float), x)
        return np.sqrt((a0 * a0) + float(self.min_v0_sq_mV2))

    def predict(
        self, freqs_hz: Any, rf_amps_mV: Any, *, clamp_x: bool = True
    ) -> np.ndarray:
        """Predict optical power at arbitrary points (broadcasted)."""
        f = np.asarray(freqs_hz, dtype=float)
        a = np.asarray(rf_amps_mV, dtype=float)
        if clamp_x:
            x = np.clip((f - float(self.freq_mid_hz)) / float(self.freq_halfspan_hz), -1.0, 1.0)
        else:
            x = (f - float(self.freq_mid_hz)) / float(self.freq_halfspan_hz)
        g = np.polyval(np.asarray(self.coeffs_g_high_to_low, dtype=float), x)
        a0 = np.polyval(np.asarray(self.coeffs_v0_a_high_to_low, dtype=float), x)
        v0 = np.sqrt((a0 * a0) + float(self.min_v0_sq_mV2))
        return g * (np.tanh(a / v0) ** 2)

    def to_aod_tanh2_calib(
        self,
        *,
        g_key: str = "*",
        v0_key: str | None = None,
        amp_scale: float,
        min_g: float = 1e-12,
        y_eps: float = 1e-6,
    ) -> AODTanh2Calib:
        """Build an `AODTanh2Calib` suitable for the synthesis pipeline."""
        if v0_key is None:
            v0_key = g_key
        return AODTanh2Calib(
            g_poly_by_logical_channel={str(g_key): tuple(float(x) for x in self.coeffs_g_high_to_low)},
            v0_a_poly_by_logical_channel={str(v0_key): tuple(float(x) for x in self.coeffs_v0_a_high_to_low)},
            freq_mid_hz=float(self.freq_mid_hz),
            freq_halfspan_hz=float(self.freq_halfspan_hz),
            amp_scale=float(amp_scale),
            min_g=float(min_g),
            min_v0_sq=float(self.min_v0_sq_mV2),
            y_eps=float(y_eps),
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation (arrays become lists)."""
        return {
            "model": "tanh2_poly",
            "degree_g": int(self.degree_g),
            "degree_v0": int(self.degree_v0),
            "freq_mid_hz": float(self.freq_mid_hz),
            "freq_halfspan_hz": float(self.freq_halfspan_hz),
            "min_v0_sq_mV2": float(self.min_v0_sq_mV2),
            "coeffs_g_high_to_low": [float(x) for x in self.coeffs_g_high_to_low],
            "coeffs_v0_a_high_to_low": [float(x) for x in self.coeffs_v0_a_high_to_low],
            "rmse": float(self.rmse),
            "max_abs_resid": float(self.max_abs_resid),
            "sse": float(self.sse),
            "opt_success": bool(self.opt_success),
            "opt_message": str(self.opt_message),
            "opt_nfev": int(self.opt_nfev),
        }


def _validate_degree(name: str, degree: int) -> int:
    d = int(degree)
    if d < 0:
        raise ValueError(f"{name} must be >= 0")
    return d


def _fit_freq_norm(curves: Sequence[OpticalPowerCalCurve]) -> tuple[float, float]:
    freqs = np.array([float(c.freq_hz) for c in curves], dtype=float)
    if freqs.size == 0:
        raise ValueError("No curves provided")
    f_min = float(np.min(freqs))
    f_max = float(np.max(freqs))
    mid = 0.5 * (f_min + f_max)
    halfspan = 0.5 * (f_max - f_min)
    if not np.isfinite(halfspan) or halfspan <= 0:
        raise ValueError("Invalid frequency span for normalization")
    return mid, halfspan


def _flatten_curves(
    curves_by_logical_channel: Mapping[str, Sequence[OpticalPowerCalCurve]] | Sequence[OpticalPowerCalCurve],
) -> list[OpticalPowerCalCurve]:
    curves: list[OpticalPowerCalCurve] = []
    if isinstance(curves_by_logical_channel, Mapping):
        for vv in curves_by_logical_channel.values():
            curves.extend(list(vv))
    else:
        curves.extend(list(curves_by_logical_channel))
    return curves


def fit_tanh2_poly_model(
    curves: Sequence[OpticalPowerCalCurve],
    *,
    degree_g: int = 6,
    degree_v0: int = 6,
    freq_mid_hz: float | None = None,
    freq_halfspan_hz: float | None = None,
    clamp_power_nonnegative: bool = True,
    min_v0_sq_mV2: float = 1e-9,
    maxiter: int = 250,
) -> Tanh2PolyFitResult:
    """
    Fit a tanh² saturation surface with frequency-dependent polynomials.

    The optimisation variable is the v0 polynomial (internally parameterized to enforce v0>0);
    the g polynomial is solved by least squares per frequency then re-fit as a polynomial each step.
    """
    dg = _validate_degree("degree_g", degree_g)
    dv0 = _validate_degree("degree_v0", degree_v0)

    min_v0_sq = float(min_v0_sq_mV2)
    if not np.isfinite(min_v0_sq) or min_v0_sq <= 0:
        raise ValueError("min_v0_sq_mV2 must be finite and > 0")

    cleaned: list[OpticalPowerCalCurve] = []
    for c in curves:
        cc = c.cleaned(clamp_power_nonnegative=clamp_power_nonnegative)
        if cc.rf_amps_mV.size > 0:
            cleaned.append(cc)
    if not cleaned:
        raise ValueError("No usable calibration points after cleaning")

    cleaned.sort(key=lambda c: float(c.freq_hz))
    freqs = np.array([float(c.freq_hz) for c in cleaned], dtype=float)
    if freq_mid_hz is None or freq_halfspan_hz is None:
        mid, halfspan = _fit_freq_norm(cleaned)
    else:
        mid = float(freq_mid_hz)
        halfspan = float(freq_halfspan_hz)
        if not np.isfinite(mid):
            raise ValueError("freq_mid_hz must be finite")
        if not np.isfinite(halfspan) or halfspan <= 0:
            raise ValueError("freq_halfspan_hz must be finite and > 0")

    x = np.clip((freqs - mid) / halfspan, -1.0, 1.0)

    a_list = [np.asarray(c.rf_amps_mV, dtype=float).reshape(-1) for c in cleaned]
    p_list = [np.asarray(c.optical_powers, dtype=float).reshape(-1) for c in cleaned]
    w_list: list[np.ndarray] = []
    for a, c in zip(a_list, cleaned, strict=True):
        if c.weights is None:
            w = np.ones_like(a, dtype=float)
        else:
            w = np.asarray(c.weights, dtype=float).reshape(-1)
        w_list.append(w)

    a_pos = [a[a > 0.0] for a in a_list]
    a_pos = [a for a in a_pos if a.size]
    if not a_pos:
        raise ValueError("Calibration data contains no positive RF amplitudes")
    a_all = np.concatenate(a_pos, axis=0)
    a_pos_min = float(np.min(a_all))
    a_pos_max = float(np.max(a_all))

    def eval_for_coeffs_v0_a(coeffs_v0_a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        coeffs_v0_a = np.asarray(coeffs_v0_a, dtype=float).reshape(-1)
        if coeffs_v0_a.shape != (dv0 + 1,):
            raise ValueError("internal shape mismatch for v0 coefficients")

        v0_a = np.polyval(coeffs_v0_a, x)
        v0 = np.sqrt((v0_a * v0_a) + min_v0_sq)  # (n_freq,)
        if not np.all(np.isfinite(v0)):
            return (
                np.full((dg + 1,), np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                float("inf"),
            )

        g_by_freq = np.zeros_like(v0, dtype=float)
        denom_by_freq = np.zeros_like(v0, dtype=float)
        for i, (a_i, p_i, w_i) in enumerate(zip(a_list, p_list, w_list, strict=True)):
            sat = np.tanh(a_i / float(v0[i])) ** 2
            sat_sq_wsum = float(np.sum(w_i * sat * sat))
            sat_sq_wsum = max(sat_sq_wsum, 1e-30)
            dot = float(np.sum(w_i * p_i * sat))
            g_by_freq[i] = dot / sat_sq_wsum
            denom_by_freq[i] = sat_sq_wsum

        w_freq = np.sqrt(np.maximum(denom_by_freq, 1e-30))
        coeffs_g = np.polyfit(x, g_by_freq, deg=dg, w=w_freq)
        g_fit = np.polyval(coeffs_g, x)

        sse = 0.0
        for i, (a_i, p_i, w_i) in enumerate(zip(a_list, p_list, w_list, strict=True)):
            sat = np.tanh(a_i / float(v0[i])) ** 2
            resid = p_i - float(g_fit[i]) * sat
            sse += float(np.sum(w_i * resid * resid))

        return np.asarray(coeffs_g, dtype=float), np.asarray(g_fit, dtype=float), np.asarray(v0, dtype=float), float(sse)

    try:
        from scipy.optimize import minimize, minimize_scalar  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for fitting optical-power calibration models") from exc

    def _initial_v0_const() -> float:
        # Search a generous range around the observed RF amplitudes.
        lo = float(np.log10(a_pos_min * 1e-3))
        hi = float(np.log10(a_pos_max * 1e3))

        def objective(log_v0: float) -> float:
            v0 = 10.0 ** float(log_v0)
            coeffs_v0_a = np.zeros((dv0 + 1,), dtype=float)
            coeffs_v0_a[-1] = float(v0)
            _coeffs_g, _g_fit, _v0, sse = eval_for_coeffs_v0_a(coeffs_v0_a)
            return float(sse)

        opt = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        return 10.0 ** float(opt.x)

    v0_init = _initial_v0_const()
    coeffs_v0_a_init = np.zeros((dv0 + 1,), dtype=float)
    coeffs_v0_a_init[-1] = float(v0_init)

    def objective_vec(coeffs_v0_a_flat: np.ndarray) -> float:
        coeffs_v0_a = np.asarray(coeffs_v0_a_flat, dtype=float).reshape(-1)
        try:
            _coeffs_g, _g_fit, _v0, sse = eval_for_coeffs_v0_a(coeffs_v0_a)
        except Exception:
            return float("inf")
        return float(sse)

    opt = minimize(
        objective_vec,
        x0=coeffs_v0_a_init,
        method="Powell",
        options={"maxiter": int(maxiter), "xtol": 1e-4, "ftol": 1e-6},
    )
    coeffs_v0_a_best = np.asarray(opt.x, dtype=float)
    coeffs_g, g_fit_by_freq, v0_by_freq, sse = eval_for_coeffs_v0_a(coeffs_v0_a_best)

    # Compute summary error metrics (unweighted, pointwise).
    resid_all: list[np.ndarray] = []
    for i, (a_i, p_i) in enumerate(zip(a_list, p_list, strict=True)):
        sat = np.tanh(a_i / float(v0_by_freq[i])) ** 2
        resid_all.append(p_i - float(g_fit_by_freq[i]) * sat)
    resid = np.concatenate(resid_all, axis=0) if resid_all else np.array([0.0], dtype=float)
    rmse = float(np.sqrt(np.mean(resid * resid)))
    max_abs = float(np.max(np.abs(resid)))

    return Tanh2PolyFitResult(
        degree_g=int(dg),
        degree_v0=int(dv0),
        freq_mid_hz=float(mid),
        freq_halfspan_hz=float(halfspan),
        min_v0_sq_mV2=float(min_v0_sq),
        coeffs_g_high_to_low=tuple(float(x) for x in np.asarray(coeffs_g, dtype=float).tolist()),
        coeffs_v0_a_high_to_low=tuple(float(x) for x in np.asarray(coeffs_v0_a_best, dtype=float).tolist()),
        freqs_hz=freqs,
        g_fit_by_freq=np.asarray(g_fit_by_freq, dtype=float),
        v0_mV_by_freq=np.asarray(v0_by_freq, dtype=float),
        rmse=rmse,
        max_abs_resid=max_abs,
        sse=float(sse),
        opt_success=bool(getattr(opt, "success", False)),
        opt_message=str(getattr(opt, "message", "")),
        opt_nfev=int(getattr(opt, "nfev", -1)),
    )


def suggest_amp_scale_from_curves(
    curves_by_logical_channel: Mapping[str, Sequence[OpticalPowerCalCurve]] | Sequence[OpticalPowerCalCurve],
) -> float:
    """Suggest an `amp_scale` that maps the max RF amplitude (mV) to ~1.0."""
    curves = _flatten_curves(curves_by_logical_channel)
    if not curves:
        raise ValueError("No curves provided")
    a_max = 0.0
    for c in curves:
        if c.rf_amps_mV.size:
            a_max = max(a_max, float(np.max(np.asarray(c.rf_amps_mV, dtype=float))))
    if not np.isfinite(a_max) or a_max <= 0:
        raise ValueError("Cannot determine max RF amplitude from curves")
    return 1.0 / float(a_max)


def fit_tanh2_poly_model_by_logical_channel(
    curves_by_logical_channel: Mapping[str, Sequence[OpticalPowerCalCurve]],
    *,
    degree_g: int = 6,
    degree_v0: int = 6,
    shared_freq_norm: bool = True,
    clamp_power_nonnegative: bool = True,
    min_v0_sq_mV2: float = 1e-9,
    maxiter: int = 250,
) -> tuple[Dict[str, Tanh2PolyFitResult], float, float]:
    """
    Fit one tanh² calibration per logical channel.

    Returns (fits_by_channel, freq_mid_hz, freq_halfspan_hz).
    If `shared_freq_norm=True`, all channel fits share the same frequency normalization.
    """
    if not curves_by_logical_channel:
        raise ValueError("curves_by_logical_channel is empty")

    all_curves: list[OpticalPowerCalCurve] = []
    for vv in curves_by_logical_channel.values():
        all_curves.extend(list(vv))
    if not all_curves:
        raise ValueError("No curves provided")

    if shared_freq_norm:
        mid, halfspan = _fit_freq_norm([c.cleaned(clamp_power_nonnegative=clamp_power_nonnegative) for c in all_curves])
    else:
        mid, halfspan = (float("nan"), float("nan"))

    fits: Dict[str, Tanh2PolyFitResult] = {}
    for lc, curves in curves_by_logical_channel.items():
        if shared_freq_norm:
            fit = fit_tanh2_poly_model(
                curves,
                degree_g=degree_g,
                degree_v0=degree_v0,
                freq_mid_hz=mid,
                freq_halfspan_hz=halfspan,
                clamp_power_nonnegative=clamp_power_nonnegative,
                min_v0_sq_mV2=min_v0_sq_mV2,
                maxiter=maxiter,
            )
        else:
            fit = fit_tanh2_poly_model(
                curves,
                degree_g=degree_g,
                degree_v0=degree_v0,
                clamp_power_nonnegative=clamp_power_nonnegative,
                min_v0_sq_mV2=min_v0_sq_mV2,
                maxiter=maxiter,
            )
        fits[str(lc)] = fit

    if shared_freq_norm:
        return fits, float(mid), float(halfspan)

    # Derive a "representative" norm from the first fit if not shared.
    first = next(iter(fits.values()))
    return fits, float(first.freq_mid_hz), float(first.freq_halfspan_hz)


def build_aod_tanh2_calib_from_fits(
    fits_by_logical_channel: Mapping[str, Tanh2PolyFitResult],
    *,
    amp_scale: float,
    min_g: float = 1e-12,
    y_eps: float = 1e-6,
    atol_freq_norm_hz: float = 1e-6,
    atol_min_v0_sq_mV2: float = 0.0,
) -> AODTanh2Calib:
    """
    Build a single `AODTanh2Calib` from per-channel fit results.

    This is convenient when one physical AOD has multiple RF channels (e.g. X/Y),
    but you want to attach a single calibration object to the program.
    """
    if not fits_by_logical_channel:
        raise ValueError("fits_by_logical_channel is empty")

    first = next(iter(fits_by_logical_channel.values()))
    mid = float(first.freq_mid_hz)
    halfspan = float(first.freq_halfspan_hz)
    min_v0_sq = float(first.min_v0_sq_mV2)

    g_poly: Dict[str, Tuple[float, ...]] = {}
    v0_poly: Dict[str, Tuple[float, ...]] = {}
    for lc, fit in fits_by_logical_channel.items():
        if not np.isclose(float(fit.freq_mid_hz), mid, rtol=0.0, atol=float(atol_freq_norm_hz)):
            raise ValueError(
                f"freq_mid_hz mismatch for {lc!r}: {fit.freq_mid_hz} vs {mid} "
                "(fit channels with shared_freq_norm=True to combine)"
            )
        if not np.isclose(
            float(fit.freq_halfspan_hz), halfspan, rtol=0.0, atol=float(atol_freq_norm_hz)
        ):
            raise ValueError(
                f"freq_halfspan_hz mismatch for {lc!r}: {fit.freq_halfspan_hz} vs {halfspan} "
                "(fit channels with shared_freq_norm=True to combine)"
            )
        if not np.isclose(
            float(fit.min_v0_sq_mV2), min_v0_sq, rtol=0.0, atol=float(atol_min_v0_sq_mV2)
        ):
            raise ValueError(
                f"min_v0_sq_mV2 mismatch for {lc!r}: {fit.min_v0_sq_mV2} vs {min_v0_sq}"
            )

        g_poly[str(lc)] = tuple(float(x) for x in fit.coeffs_g_high_to_low)
        v0_poly[str(lc)] = tuple(float(x) for x in fit.coeffs_v0_a_high_to_low)

    return AODTanh2Calib(
        g_poly_by_logical_channel=g_poly,
        v0_a_poly_by_logical_channel=v0_poly,
        freq_mid_hz=mid,
        freq_halfspan_hz=halfspan,
        amp_scale=float(amp_scale),
        min_g=float(min_g),
        min_v0_sq=float(min_v0_sq),
        y_eps=float(y_eps),
    )


def aod_tanh2_calib_to_python(
    calib: AODTanh2Calib, *, var_name: str = "AOD_TANH2_CALIB"
) -> str:
    """Format an `AODTanh2Calib` as a Python constant snippet."""
    def _fmt_tuple(vals: Iterable[float]) -> str:
        vv = tuple(float(v) for v in vals)
        suffix = "," if len(vv) == 1 else ""
        return "(" + ", ".join(f"{v:.16g}" for v in vv) + suffix + ")"

    g_lines = []
    for k in sorted(calib.g_poly_by_logical_channel.keys()):
        g_lines.append(f"        {k!r}: {_fmt_tuple(calib.g_poly_by_logical_channel[k])},")
    v_lines = []
    for k in sorted(calib.v0_a_poly_by_logical_channel.keys()):
        v_lines.append(f"        {k!r}: {_fmt_tuple(calib.v0_a_poly_by_logical_channel[k])},")

    return "\n".join(
        [
            f"{var_name} = AODTanh2Calib(",
            "    g_poly_by_logical_channel={",
            *g_lines,
            "    },",
            "    v0_a_poly_by_logical_channel={",
            *v_lines,
            "    },",
            f"    freq_mid_hz={float(calib.freq_mid_hz):.16g},",
            f"    freq_halfspan_hz={float(calib.freq_halfspan_hz):.16g},",
            f"    amp_scale={float(calib.amp_scale):.16g},",
            f"    min_g={float(calib.min_g):.16g},",
            f"    min_v0_sq={float(calib.min_v0_sq):.16g},",
            f"    y_eps={float(calib.y_eps):.16g},",
            ")",
        ]
    )
