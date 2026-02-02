"""
Example: plot a "DE compensation" calibration file (JSON).

This example is designed for calibration files like:
  `examples/814_H_calFile_17.02.2022_0=0.txt`

Even though the extension is `.txt`, the contents are a single JSON object.

--- Data format (top-level JSON keys) ---

The file is a JSON dict with keys:

1) `umPerMHz` (float)
   A simple conversion factor (µm / MHz). This is often used to convert an AOD
   frequency difference into a position shift.

2) `DE_RF_calibration` (dict)
   Maps frequency (MHz) -> a measured curve of diffraction efficiency vs RF drive.

   Structure:
     DE_RF_calibration["100.13"] = {
         "RF Amplitude (mV)": [a0, a1, ...],
         "Diffraction Efficiency": [de0, de1, ...],
     }

   Notes:
   - JSON object keys are always strings, so frequencies are stored as strings.
   - The RF amplitude grid is the same for all frequencies in this file.
   - "Diffraction Efficiency" is unitless and may contain slight negative values
     (noise) or values > 1.0 depending on normalization.

3) `Power_calibration` (dict)
   Maps target optical power (0..1, stored as a string key) -> required RF amplitude
   vs frequency.

   Structure:
     Power_calibration["0.25"] = {
         "Frequency (MHz)": [f0, f1, ...],
         "RF Amplitude (mV)": [a0, a1, ...],
     }

   Notes:
   - Each entry is a set of (freq, rf_amp) points at that target optical power.
   - Frequency sampling can be irregular and may include repeated frequencies; this
     example averages duplicates per power slice before plotting.

--- What this script plots ---

Figure 1:
  - Left: a 2D heatmap of `DE_RF_calibration` (x=freq MHz, y=RF amp mV, color=DE).
  - Right: a 2D contour plot of `Power_calibration` (x=freq MHz, y=target power, color=RF amp mV).

Figure 2:
  - Left: a slice through `DE_RF_calibration` at the centre frequency (DE vs RF amp).
  - Right: a slice through `DE_RF_calibration` at the centre RF amp (DE vs frequency).

Figure 3:
  - Data/model/residual heatmaps for a simple analytic fit of `DE_RF_calibration`:
      DE(freq, amp) ≈ g(freq) * (amp^2 / (amp^2 + u0(freq)))
    where `g(freq)` is a 6th-order polynomial (in normalized frequency) and `u0(freq)` is a
    frequency-dependent saturation parameter (also a 6th-order polynomial, enforced positive).
    Optionally, you can add a small "quartic" correction while preserving normalization:
      DE(freq, amp) ≈ g(freq) * sat_mix(amp; u0(freq), w, b)
      sat_mix = (1-w) * (a^2 / (a^2 + u0)) + w * (a^4 / (a^4 + b))
    with `0 <= w <= 1` and `b > 0`. This remains monotonic in `a >= 0` and is invertible
    (in `u=a^2`, the inverse reduces to a cubic equation).
    Alternatively, you can use a smooth "monotone Bragg-like" saturation:
      DE(freq, amp) ≈ g(freq) * tanh^2(a / v0(freq))

Figure 4:
  - Residual slices (data - fit) at the same centre (freq, amp) points as Figure 2.

Usage:
  python examples/plot_de_compensation_file.py
  python examples/plot_de_compensation_file.py path/to/calFile.txt
  python examples/plot_de_compensation_file.py --sat-model tanh2
  python examples/plot_de_compensation_file.py --sat-model mix_quartic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(cal: dict) -> None:
    keys = list(cal.keys())
    print("--- calibration file summary ---")
    print("top-level keys:", keys)

    um_per_mhz = float(cal.get("umPerMHz", float("nan")))
    print("umPerMHz:", um_per_mhz)

    de_rf = cal.get("DE_RF_calibration", {})
    if isinstance(de_rf, dict) and de_rf:
        freqs = sorted(float(k) for k in de_rf.keys())
        ex = de_rf[str(freqs[len(freqs) // 2])]
        amps = ex.get("RF Amplitude (mV)", [])
        print("DE_RF_calibration: n_freqs =", len(freqs), "freq range =", (freqs[0], freqs[-1]))
        print("  per-frequency curve length =", len(amps), "RF amp range =", (min(amps), max(amps)))
    else:
        print("DE_RF_calibration: missing or empty")

    power_cal = cal.get("Power_calibration", {})
    if isinstance(power_cal, dict) and power_cal:
        powers = sorted(float(k) for k in power_cal.keys())
        print(
            "Power_calibration: n_powers =",
            len(powers),
            "power range =",
            (powers[0], powers[-1]),
        )
        # Count points without materializing all arrays.
        n_points = 0
        f_min = float("inf")
        f_max = -float("inf")
        for v in power_cal.values():
            ff = v.get("Frequency (MHz)", [])
            if not ff:
                continue
            n_points += len(ff)
            f_min = min(f_min, float(min(ff)))
            f_max = max(f_max, float(max(ff)))
        print("  total (freq,power) points =", n_points, "freq range =", (f_min, f_max))
    else:
        print("Power_calibration: missing or empty")


def _de_rf_grid(de_rf: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort frequency keys numerically (keys are strings).
    freq_keys = sorted(de_rf.keys(), key=lambda k: float(k))
    freqs_mhz = np.array([float(k) for k in freq_keys], dtype=float)

    # RF amplitude grid is identical for all frequencies in this file.
    amps_mV = np.asarray(de_rf[freq_keys[0]]["RF Amplitude (mV)"], dtype=float)

    de = np.empty((len(freq_keys), len(amps_mV)), dtype=float)  # (n_freq, n_amp)
    for i, k in enumerate(freq_keys):
        row = np.asarray(de_rf[k]["Diffraction Efficiency"], dtype=float)
        if row.shape != amps_mV.shape:
            raise ValueError(
                f"DE_RF_calibration[{k!r}]: expected {amps_mV.shape} values, got {row.shape}"
            )
        de[i, :] = row
    return freqs_mhz, amps_mV, de


def _plot_de_rf(ax, *, freqs_mhz: np.ndarray, amps_mV: np.ndarray, de: np.ndarray) -> None:
    if de.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError(
            f"DE grid shape mismatch: de={de.shape}, freqs={freqs_mhz.shape}, amps={amps_mV.shape}"
        )

    # Plot with pcolormesh: x=freq, y=amp, C=(n_amp, n_freq)
    pcm = ax.pcolormesh(freqs_mhz, amps_mV, de.T, shading="auto", cmap="viridis")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("RF amplitude (mV)")
    ax.set_title("DE_RF_calibration: Diffraction efficiency")
    import matplotlib.pyplot as plt

    plt.colorbar(pcm, ax=ax, label="Diffraction efficiency (arb)")


def _plot_de_rf_slices(
    ax_amp,
    ax_freq,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de: np.ndarray,
    de_model: np.ndarray | None = None,
) -> tuple[float, float]:
    # Pick "centre of range" (midpoint of min/max) and snap to nearest grid points.
    f0 = float(freqs_mhz.min())
    f1 = float(freqs_mhz.max())
    a0 = float(amps_mV.min())
    a1 = float(amps_mV.max())
    f_target = 0.5 * (f0 + f1)
    a_target = 0.5 * (a0 + a1)
    i_f = int(np.argmin(np.abs(freqs_mhz - f_target)))
    i_a = int(np.argmin(np.abs(amps_mV - a_target)))
    f_sel = float(freqs_mhz[i_f])
    a_sel = float(amps_mV[i_a])

    # Slice along RF amplitude axis (hold frequency constant).
    ax_amp.plot(amps_mV, de[i_f, :], lw=1.5, label="data")
    if de_model is not None:
        ax_amp.plot(amps_mV, de_model[i_f, :], lw=1.5, ls="--", label="model")
    ax_amp.set_xlabel("RF amplitude (mV)")
    ax_amp.set_ylabel("Diffraction efficiency (arb)")
    ax_amp.set_title(f"DE vs RF amplitude @ {f_sel:.2f} MHz")
    ax_amp.grid(True, alpha=0.25)
    if de_model is not None:
        ax_amp.legend(loc="best")

    # Slice along frequency axis (hold RF amplitude constant).
    ax_freq.plot(freqs_mhz, de[:, i_a], lw=1.5, label="data")
    if de_model is not None:
        ax_freq.plot(freqs_mhz, de_model[:, i_a], lw=1.5, ls="--", label="model")
    ax_freq.set_xlabel("RF frequency (MHz)")
    ax_freq.set_ylabel("Diffraction efficiency (arb)")
    ax_freq.set_title(f"DE vs frequency @ {a_sel:.2f} mV")
    ax_freq.grid(True, alpha=0.25)
    if de_model is not None:
        ax_freq.legend(loc="best")

    return f_sel, a_sel


def _plot_power_cal(ax, power_cal: dict) -> None:
    # Build scattered points (freq_mhz, power, rf_amp_mV).
    fs: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    amps: list[np.ndarray] = []

    power_items = sorted(power_cal.items(), key=lambda kv: float(kv[0]))
    for p_str, entry in power_items:
        p = float(p_str)
        f = np.asarray(entry["Frequency (MHz)"], dtype=float).reshape(-1)
        a = np.asarray(entry["RF Amplitude (mV)"], dtype=float).reshape(-1)
        if f.shape != a.shape:
            raise ValueError(
                f"Power_calibration[{p_str!r}]: Frequency and RF Amplitude lengths differ "
                f"({f.shape} vs {a.shape})"
            )
        m = np.isfinite(f) & np.isfinite(a)
        f = f[m]
        a = a[m]
        if f.size == 0:
            continue

        # De-duplicate repeated frequencies within the slice (average amplitudes).
        # This avoids QHull issues when triangulating for contour plots.
        uniq_f, inv = np.unique(f, return_inverse=True)
        a_sum = np.zeros_like(uniq_f, dtype=float)
        counts = np.zeros_like(uniq_f, dtype=int)
        np.add.at(a_sum, inv, a)
        np.add.at(counts, inv, 1)
        a_mean = a_sum / counts

        fs.append(uniq_f)
        amps.append(a_mean)
        ps.append(np.full_like(uniq_f, p, dtype=float))

    F = np.concatenate(fs, axis=0)
    P = np.concatenate(ps, axis=0)
    A = np.concatenate(amps, axis=0)

    # Irregular sampling -> triangulated contours.
    tcf = ax.tricontourf(F, P, A, levels=60, cmap="viridis")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("Target optical power (0..1)")
    ax.set_title("Power_calibration: required RF amplitude")

    import matplotlib.pyplot as plt

    plt.colorbar(tcf, ax=ax, label="RF amplitude (mV)")


def _polyval_horner(coeffs_high_to_low: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.asarray(coeffs_high_to_low, dtype=float).reshape(-1)
    if c.size == 0:
        return np.zeros_like(x, dtype=float)
    y = np.zeros_like(x, dtype=float) + float(c[0])
    for cc in c[1:]:
        y = y * x + float(cc)
    return y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # Clip to keep exp() finite in extreme optimizer steps.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = float(p)
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        raise ValueError("logit() expects p in (0, 1)")
    return float(np.log(p / (1.0 - p)))


def _sat_u_over_u_plus_u0(u: np.ndarray, u0_by_freq: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    u0_by_freq = np.asarray(u0_by_freq, dtype=float).reshape(-1)
    return u[None, :] / (u[None, :] + u0_by_freq[:, None])


def _sat_tanh2(a_mV: np.ndarray, v0_mV_by_freq: np.ndarray) -> np.ndarray:
    """
    Smooth, monotone saturation curve:
      sat(a) = tanh^2(a / v0)

    This has small-signal behavior sat ~ (a/v0)^2 and saturates to 1 at large a.
    """
    a = np.asarray(a_mV, dtype=float).reshape(-1)
    v0 = np.asarray(v0_mV_by_freq, dtype=float).reshape(-1)
    if not np.all(np.isfinite(v0)):
        raise ValueError("v0_mV_by_freq must be finite")
    if np.any(v0 <= 0):
        raise ValueError("v0_mV_by_freq must be > 0")
    t = a[None, :] / v0[:, None]
    return np.tanh(t) ** 2


def _sat_mix_quartic(
    u: np.ndarray,
    u0_by_freq: np.ndarray,
    *,
    w: float,
    b_mV4: float,
) -> np.ndarray:
    """
    Normalized mixture of a quadratic and quartic saturation curve.

    In RF amplitude `a` (mV), we use u=a^2 and:
      sat_mix = (1-w) * u/(u+u0) + w * u^2/(u^2 + b)
    where `b` has units of mV^4 (since u^2 = a^4).
    """
    w = float(w)
    if not np.isfinite(w) or w < 0.0 or w > 1.0:
        raise ValueError("w must be finite and in [0, 1]")
    b = float(b_mV4)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("b_mV4 must be finite and > 0")

    u = np.asarray(u, dtype=float).reshape(-1)
    u2 = u * u
    u0_by_freq = np.asarray(u0_by_freq, dtype=float).reshape(-1)

    # sat2 is frequency-independent, but return it with the same (n_freq, n_amp) shape.
    sat2_1 = u2[None, :] / (u2[None, :] + b)  # (1, n_amp)
    sat2 = np.broadcast_to(sat2_1, (u0_by_freq.size, u.size))
    if w == 1.0:
        return sat2
    sat1 = _sat_u_over_u_plus_u0(u, u0_by_freq)
    if w == 0.0:
        return sat1
    return (1.0 - w) * sat1 + w * sat2


def _fit_sat_poly_model(
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de: np.ndarray,
    degree_g: int = 6,
    degree_knee: int = 6,
    sat_model: str = "u_over_u_plus_u0",
    # Only used for sat_model == "mix_quartic".
    quartic_w_init: float = 0.02,
    quartic_b_init_mV4: float | None = None,
    clamp_de_nonnegative: bool = True,
    min_u0_mV2: float = 1e-9,
) -> dict[str, object]:
    """
    Fit:
        DE(freq, amp) ≈ g(freq) * sat(amp^2; ...)

        sat_model="u_over_u_plus_u0":
            sat(u; u0) = u / (u + u0)

        sat_model="tanh2":
            sat(a; v0) = tanh^2(a / v0)
            where `v0>0` has units of mV.

        sat_model="mix_quartic":
            sat_mix(u; u0, w, b) = (1-w) * u/(u+u0) + w * u^2/(u^2 + b)
            where `0<=w<=1` and `b>0` (units: mV^4).
        g(freq) is a degree-`degree_g` polynomial in normalized frequency x ∈ [-1, 1].
        u0(freq) is a degree-`degree_knee` polynomial in normalized frequency, squared to enforce u0>0.

    Returns fit parameters plus convenience arrays for plotting.
    """
    if degree_g < 0:
        raise ValueError("degree_g must be >= 0")
    if degree_knee < 0:
        raise ValueError("degree_knee must be >= 0")
    if de.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError("shape mismatch for freqs/amps/de")

    sat_model = str(sat_model).strip()
    if sat_model not in {"u_over_u_plus_u0", "tanh2", "mix_quartic"}:
        raise ValueError("sat_model must be one of {'u_over_u_plus_u0', 'tanh2', 'mix_quartic'}")

    min_u0 = float(min_u0_mV2)
    if not np.isfinite(min_u0) or min_u0 <= 0:
        raise ValueError("min_u0_mV2 must be finite and > 0")

    de_raw = np.asarray(de, dtype=float)
    if clamp_de_nonnegative:
        de_used = np.clip(de_raw, 0.0, None)
    else:
        de_used = de_raw

    # Normalize frequency to x ∈ [-1, 1] for a well-conditioned polynomial.
    f_min = float(np.min(freqs_mhz))
    f_max = float(np.max(freqs_mhz))
    f_mid = 0.5 * (f_min + f_max)
    f_halfspan = 0.5 * (f_max - f_min)
    if not np.isfinite(f_halfspan) or f_halfspan <= 0:
        raise ValueError("invalid frequency range for normalization")
    x = (np.asarray(freqs_mhz, dtype=float) - f_mid) / f_halfspan

    a = np.asarray(amps_mV, dtype=float).reshape(-1)
    u = a * a
    u_pos = u[u > 0]
    if u_pos.size == 0:
        raise ValueError("RF amplitude grid has no positive values")

    # A useful scale for initializing the knee.
    u_pos_min = float(np.min(u_pos))
    u_pos_max = float(np.max(u_pos))
    a_pos_min = float(np.min(a[a > 0]))
    a_pos_max = float(np.max(a))

    de_sq_sum = float(np.sum(de_used * de_used))

    def _a0_and_u0_for_coeffs(coeffs_a0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a0 = _polyval_horner(np.asarray(coeffs_a0, dtype=float), x)
        u0 = (a0 * a0) + min_u0
        return a0, u0

    def _a0_and_v0_for_coeffs(coeffs_a0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a0 = _polyval_horner(np.asarray(coeffs_a0, dtype=float), x)
        v0 = np.sqrt((a0 * a0) + min_u0)
        return a0, v0

    def eval_for_coeffs_a0(
        coeffs_a0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        a0, u0 = _a0_and_u0_for_coeffs(coeffs_a0)
        if not np.all(np.isfinite(u0)):
            return (
                np.full((degree_g + 1,), np.nan),
                np.full((degree_knee + 1,), np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                float("inf"),
            )

        # sat is per-frequency because u0 varies with freq.
        sat = _sat_u_over_u_plus_u0(u, u0)  # (n_freq, n_amp)
        denom = np.sum(sat * sat, axis=1)  # (n_freq,)
        # Avoid divide-by-zero: if denom is zero, sat is all-zero -> data must also be all-zero.
        denom = np.maximum(denom, 1e-30)
        dots = np.sum(de_used * sat, axis=1)  # (n_freq,)
        g = dots / denom

        # Polynomial fit for g(freq): coefficients high->low (np.polyfit convention).
        coeffs_g = np.polyfit(x, g, deg=degree_g)
        g_fit = _polyval_horner(coeffs_g, x)

        # SSE computed without materializing the full (n_freq,n_amp) fit matrix:
        # sum_j (de - g*sat)^2 = sum(de^2) - 2*g*dot + g^2*denom
        sse = de_sq_sum - (2.0 * float(np.dot(g_fit, dots))) + float(np.dot((g_fit * g_fit), denom))
        return (
            np.asarray(coeffs_g, dtype=float),
            np.asarray(coeffs_a0, dtype=float),
            np.asarray(a0, dtype=float),
            np.asarray(u0, dtype=float),
            np.asarray(g_fit, dtype=float),
            float(sse),
        )

    def eval_for_coeffs_a0_tanh2(
        coeffs_a0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        a0, v0 = _a0_and_v0_for_coeffs(coeffs_a0)
        if not np.all(np.isfinite(v0)):
            return (
                np.full((degree_g + 1,), np.nan),
                np.full((degree_knee + 1,), np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                float("inf"),
            )

        sat = _sat_tanh2(a, v0)  # (n_freq, n_amp)
        denom = np.sum(sat * sat, axis=1)  # (n_freq,)
        denom = np.maximum(denom, 1e-30)
        dots = np.sum(de_used * sat, axis=1)  # (n_freq,)
        g = dots / denom
        coeffs_g = np.polyfit(x, g, deg=degree_g)
        g_fit = _polyval_horner(coeffs_g, x)
        sse = de_sq_sum - (2.0 * float(np.dot(g_fit, dots))) + float(np.dot((g_fit * g_fit), denom))
        return (
            np.asarray(coeffs_g, dtype=float),
            np.asarray(coeffs_a0, dtype=float),
            np.asarray(a0, dtype=float),
            np.asarray(v0, dtype=float),
            np.asarray(g_fit, dtype=float),
            float(sse),
        )

    def eval_for_coeffs_a0_and_quartic(
        coeffs_a0: np.ndarray, *, w: float, b_mV4: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        a0, u0 = _a0_and_u0_for_coeffs(coeffs_a0)
        if not np.all(np.isfinite(u0)):
            return (
                np.full((degree_g + 1,), np.nan),
                np.full((degree_knee + 1,), np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                np.full_like(x, np.nan),
                float("inf"),
            )

        sat = _sat_mix_quartic(u, u0, w=float(w), b_mV4=float(b_mV4))
        denom = np.sum(sat * sat, axis=1)  # (n_freq,)
        denom = np.maximum(denom, 1e-30)
        dots = np.sum(de_used * sat, axis=1)  # (n_freq,)
        g = dots / denom
        coeffs_g = np.polyfit(x, g, deg=degree_g)
        g_fit = _polyval_horner(coeffs_g, x)
        sse = de_sq_sum - (2.0 * float(np.dot(g_fit, dots))) + float(np.dot((g_fit * g_fit), denom))
        return (
            np.asarray(coeffs_g, dtype=float),
            np.asarray(coeffs_a0, dtype=float),
            np.asarray(a0, dtype=float),
            np.asarray(u0, dtype=float),
            np.asarray(g_fit, dtype=float),
            float(sse),
        )

    try:
        from scipy.optimize import minimize, minimize_scalar  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for fitting the analytic model") from exc

    # Initialize:
    # - If u0 is constant, solve the 1D problem for a strong initial guess.
    # - Then use a frequency-dependent knee polynomial initialized to the same constant.
    def _initial_knee_const() -> float:
        if sat_model == "tanh2":
            lo = float(np.log10(a_pos_min * 1e-3))
            hi = float(np.log10(a_pos_max * 1e3))

            def objective(log_v0: float) -> float:
                v0 = 10.0 ** float(log_v0)
                coeffs_a0 = np.zeros((degree_knee + 1,), dtype=float)
                coeffs_a0[-1] = float(v0)
                _coeffs_g, _coeffs_a0, _a0, _v0, _g_fit, sse = eval_for_coeffs_a0_tanh2(coeffs_a0)
                return sse

            opt = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
            return 10.0 ** float(opt.x)

        lo = float(np.log10(u_pos_min * 1e-3))
        hi = float(np.log10(u_pos_max * 1e3))

        def objective(log_u0: float) -> float:
            u0 = 10.0 ** float(log_u0)
            # Reuse the general evaluator with a constant a0.
            a0_const = float(np.sqrt(max(u0, min_u0)))
            coeffs_a0 = np.zeros((degree_knee + 1,), dtype=float)
            coeffs_a0[-1] = a0_const
            _coeffs_g, _coeffs_a0, _a0, _u0, _g_fit, sse = eval_for_coeffs_a0(coeffs_a0)
            return sse

        opt = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        return float(np.sqrt(max(10.0 ** float(opt.x), min_u0)))

    knee_init_mV = _initial_knee_const()
    u0_init_mV2 = float(max(knee_init_mV * knee_init_mV, min_u0))
    a0_init = float(knee_init_mV)
    coeffs_a0_init = np.zeros((degree_knee + 1,), dtype=float)
    coeffs_a0_init[-1] = a0_init

    def objective_vec(p_flat: np.ndarray) -> float:
        p = np.asarray(p_flat, dtype=float).reshape(-1)
        if sat_model == "u_over_u_plus_u0":
            coeffs_a0 = p
            if coeffs_a0.shape != (degree_knee + 1,):
                raise ValueError("internal shape mismatch for knee coefficients")
            _coeffs_g, _coeffs_a0, _a0, _u0, _g_fit, sse = eval_for_coeffs_a0(coeffs_a0)
            return float(sse)

        if sat_model == "tanh2":
            coeffs_a0 = p
            if coeffs_a0.shape != (degree_knee + 1,):
                raise ValueError("internal shape mismatch for knee coefficients")
            _coeffs_g, _coeffs_a0, _a0, _v0, _g_fit, sse = eval_for_coeffs_a0_tanh2(coeffs_a0)
            return float(sse)

        if p.size != (degree_knee + 1) + 2:
            raise ValueError("internal shape mismatch for quartic parameter vector")
        coeffs_a0 = p[: degree_knee + 1]
        w = float(_sigmoid(p[degree_knee + 1]))
        log_b = float(p[degree_knee + 2])
        b_mV4 = float(np.exp(log_b))
        try:
            _coeffs_g, _coeffs_a0, _a0, _u0, _g_fit, sse = eval_for_coeffs_a0_and_quartic(
                coeffs_a0, w=w, b_mV4=b_mV4
            )
        except Exception:
            return float("inf")
        return float(sse)

    if sat_model in {"u_over_u_plus_u0", "tanh2"}:
        p0 = coeffs_a0_init
    else:  # mix_quartic
        w0 = float(quartic_w_init)
        w0 = float(np.clip(w0, 1e-6, 1.0 - 1e-6))
        b0 = (
            float(quartic_b_init_mV4)
            if quartic_b_init_mV4 is not None
            else float(u0_init_mV2 * u0_init_mV2)
        )
        b0 = float(max(b0, min_u0 * min_u0))
        p0 = np.concatenate(
            [
                coeffs_a0_init,
                np.array([_logit(w0), float(np.log(b0))], dtype=float),
            ],
            axis=0,
        )

    opt = minimize(
        objective_vec,
        x0=p0,
        method="Powell",
        options={"maxiter": 250, "xtol": 1e-4, "ftol": 1e-6},
    )
    p_best = np.asarray(opt.x, dtype=float)

    quartic_w: float | None = None
    quartic_b_mV4: float | None = None
    v0_by_freq: np.ndarray | None = None
    if sat_model == "u_over_u_plus_u0":
        coeffs_a0_best = p_best
        coeffs_g, _coeffs_a0, a0_by_freq, u0_by_freq, g_fit, sse = eval_for_coeffs_a0(coeffs_a0_best)
        sat = _sat_u_over_u_plus_u0(u, u0_by_freq)
    elif sat_model == "tanh2":
        coeffs_a0_best = p_best
        coeffs_g, _coeffs_a0, a0_by_freq, v0_by_freq, g_fit, sse = eval_for_coeffs_a0_tanh2(coeffs_a0_best)
        sat = _sat_tanh2(a, v0_by_freq)
        u0_by_freq = (v0_by_freq * v0_by_freq)  # for consistent downstream reporting
    else:
        coeffs_a0_best = p_best[: degree_knee + 1]
        quartic_w = float(_sigmoid(p_best[degree_knee + 1]))
        quartic_b_mV4 = float(np.exp(p_best[degree_knee + 2]))
        coeffs_g, _coeffs_a0, a0_by_freq, u0_by_freq, g_fit, sse = eval_for_coeffs_a0_and_quartic(
            coeffs_a0_best, w=quartic_w, b_mV4=quartic_b_mV4
        )
        sat = _sat_mix_quartic(u, u0_by_freq, w=quartic_w, b_mV4=quartic_b_mV4)

    # Build the fitted surface for residual plots.
    de_fit = g_fit[:, None] * sat

    rmse = float(np.sqrt(np.mean((de_used - de_fit) ** 2)))
    max_abs = float(np.max(np.abs(de_used - de_fit)))

    return {
        "degree_g": int(degree_g),
        "degree_knee": int(degree_knee),
        "sat_model": sat_model,
        "quartic_w": quartic_w,
        "quartic_b_mV4": quartic_b_mV4,
        "v0_mV_by_freq": v0_by_freq,
        "min_u0_mV2": float(min_u0),
        "u0_mV2_by_freq": u0_by_freq,
        "a0_mV_by_freq": a0_by_freq,
        "freq_mid_mhz": float(f_mid),
        "freq_halfspan_mhz": float(f_halfspan),
        "x": x,
        "coeffs_g_high_to_low": np.asarray(coeffs_g, dtype=float),
        "coeffs_a0_high_to_low": np.asarray(coeffs_a0_best, dtype=float),
        "g_fit_by_freq": np.asarray(g_fit, dtype=float),
        "sat_by_amp": np.asarray(sat, dtype=float),
        "de_raw": de_raw,
        "de_used": de_used,
        "de_fit": np.asarray(de_fit, dtype=float),
        "sse": float(sse),
        "rmse": rmse,
        "max_abs_resid": max_abs,
        "opt_status": str(getattr(opt, "message", "")),
        "opt_success": bool(getattr(opt, "success", False)),
        "opt_nfev": int(getattr(opt, "nfev", -1)),
    }


def _plot_fit_and_residuals(
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    fit: dict[str, object],
):
    import matplotlib.pyplot as plt

    de_used = np.asarray(fit["de_used"], dtype=float)
    de_fit = np.asarray(fit["de_fit"], dtype=float)
    resid = de_used - de_fit

    vmin = float(np.nanmin(de_used))
    vmax = float(np.nanmax(de_used))
    rmax = float(np.nanmax(np.abs(resid)))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharex=False, sharey=False)
    ax0, ax1, ax2 = axs

    pcm0 = ax0.pcolormesh(freqs_mhz, amps_mV, de_used.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.set_title("DE data (clipped >= 0)")
    ax0.set_xlabel("RF frequency (MHz)")
    ax0.set_ylabel("RF amplitude (mV)")
    plt.colorbar(pcm0, ax=ax0, label="DE (arb)")

    pcm1 = ax1.pcolormesh(freqs_mhz, amps_mV, de_fit.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("Analytic fit")
    ax1.set_xlabel("RF frequency (MHz)")
    ax1.set_ylabel("RF amplitude (mV)")
    plt.colorbar(pcm1, ax=ax1, label="DE (arb)")

    pcm2 = ax2.pcolormesh(
        freqs_mhz,
        amps_mV,
        resid.T,
        shading="auto",
        cmap="coolwarm",
        vmin=-rmax,
        vmax=rmax,
    )
    ax2.set_title("Residual (data - fit)")
    ax2.set_xlabel("RF frequency (MHz)")
    ax2.set_ylabel("RF amplitude (mV)")
    plt.colorbar(pcm2, ax=ax2, label="Residual (arb)")

    coeffs_g = np.asarray(fit["coeffs_g_high_to_low"], dtype=float)
    coeffs_a0 = np.asarray(fit["coeffs_a0_high_to_low"], dtype=float)
    a0_by_f = np.asarray(fit["a0_mV_by_freq"], dtype=float).reshape(-1)
    u0_by_f = np.asarray(fit["u0_mV2_by_freq"], dtype=float).reshape(-1)
    rmse = float(fit["rmse"])
    max_abs = float(fit["max_abs_resid"])
    sat_model = str(fit.get("sat_model", "u_over_u_plus_u0"))
    quartic_w = fit.get("quartic_w")
    quartic_b = fit.get("quartic_b_mV4")

    if sat_model == "mix_quartic":
        model_str = "DE ≈ g(x) * [(1-w)*a²/(a²+u0(x)) + w*a⁴/(a⁴+b)]"
        extra = f"   w={float(quartic_w):.4g}, b={float(quartic_b):.4g} mV⁴"
    elif sat_model == "tanh2":
        model_str = "DE ≈ g(x) * tanh²(a/v0(x))"
        extra = ""
    else:
        model_str = "DE ≈ g(x) * (a²/(a²+u0(x)))"
        extra = ""
    fig.suptitle(
        f"Fit: {model_str}   RMSE={rmse:.4g}, max|res|={max_abs:.4g}{extra}"
    )
    fig.tight_layout()

    print("\n--- analytic fit (DE_RF_calibration) ---")
    if sat_model == "mix_quartic":
        print("Model: DE(freq, a) = g(freq) * [(1-w)*a^2/(a^2+u0(freq)) + w*a^4/(a^4+b)]")
        print("  w =", float(quartic_w))
        print("  b_mV^4 =", float(quartic_b))
    elif sat_model == "tanh2":
        print("Model: DE(freq, a) = g(freq) * tanh^2(a / v0(freq))")
    else:
        print("Model: DE(freq, a) = g(freq) * (a^2 / (a^2 + u0(freq)))")
    print("  freq normalization: x = (f_MHz - mid)/halfspan")
    print("  mid_MHz =", float(fit["freq_mid_mhz"]))
    print("  halfspan_MHz =", float(fit["freq_halfspan_mhz"]))
    print("  g(x) poly coeffs (high->low) =", coeffs_g.tolist())
    print("  a0(x) poly coeffs (high->low) =", coeffs_a0.tolist())
    if sat_model == "tanh2":
        v0_by_f = np.asarray(fit["v0_mV_by_freq"], dtype=float).reshape(-1)
        print("  v0_mV range =", (float(np.min(v0_by_f)), float(np.max(v0_by_f))))
    else:
        print(
            "  a0_mV range =",
            (float(np.min(a0_by_f)), float(np.max(a0_by_f))),
            " (u0_mV^2 range =",
            (float(np.min(u0_by_f)), float(np.max(u0_by_f))),
            ")",
        )
    print("  RMSE =", rmse)
    print("  max|residual| =", max_abs)

    return fig


def _plot_residual_slices(
    ax_amp,
    ax_freq,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    resid: np.ndarray,
    f_sel_mhz: float,
    a_sel_mV: float,
) -> None:
    if resid.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError("residual grid shape mismatch")

    i_f = int(np.argmin(np.abs(freqs_mhz - float(f_sel_mhz))))
    i_a = int(np.argmin(np.abs(amps_mV - float(a_sel_mV))))
    f_sel = float(freqs_mhz[i_f])
    a_sel = float(amps_mV[i_a])

    ax_amp.plot(amps_mV, resid[i_f, :], lw=1.5)
    ax_amp.axhline(0.0, color="k", lw=1.0, alpha=0.35)
    ax_amp.set_xlabel("RF amplitude (mV)")
    ax_amp.set_ylabel("Residual (arb)")
    ax_amp.set_title(f"Residual vs RF amplitude @ {f_sel:.2f} MHz")
    ax_amp.grid(True, alpha=0.25)

    ax_freq.plot(freqs_mhz, resid[:, i_a], lw=1.5)
    ax_freq.axhline(0.0, color="k", lw=1.0, alpha=0.35)
    ax_freq.set_xlabel("RF frequency (MHz)")
    ax_freq.set_ylabel("Residual (arb)")
    ax_freq.set_title(f"Residual vs frequency @ {a_sel:.2f} mV")
    ax_freq.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="examples/814_H_calFile_17.02.2022_0=0.txt",
        help="Path to a DE compensation calibration JSON file.",
    )
    parser.add_argument(
        "--sat-model",
        choices=["u_over_u_plus_u0", "tanh2", "mix_quartic"],
        default="u_over_u_plus_u0",
        help="Which saturation curve to fit.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    cal = _load_json(path)
    _summarize(cal)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install the `dev` dependency group."
        ) from exc

    de_rf = cal.get("DE_RF_calibration")
    power_cal = cal.get("Power_calibration")
    if not isinstance(de_rf, dict) or not isinstance(power_cal, dict):
        raise ValueError(
            "Expected JSON keys 'DE_RF_calibration' and 'Power_calibration' to be dicts."
        )

    freqs_mhz, amps_mV, de = _de_rf_grid(de_rf)
    fit = _fit_sat_poly_model(
        freqs_mhz=freqs_mhz,
        amps_mV=amps_mV,
        de=de,
        degree_g=6,
        degree_knee=6,
        sat_model=str(args.sat_model),
    )
    de_fit = np.asarray(fit["de_fit"], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)
    _plot_de_rf(ax0, freqs_mhz=freqs_mhz, amps_mV=amps_mV, de=de)
    _plot_power_cal(ax1, power_cal)
    fig.suptitle(path.name)
    fig.tight_layout()

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 4), sharex=False, sharey=False)
    f_sel, a_sel = _plot_de_rf_slices(
        ax2, ax3, freqs_mhz=freqs_mhz, amps_mV=amps_mV, de=de, de_model=de_fit
    )
    fig2.suptitle(f"{path.name} (slices at centre of ranges: f={f_sel:.2f} MHz, a={a_sel:.2f} mV)")
    fig2.tight_layout()

    # Mark slice positions on the DE heatmap (figure 1, left axis).
    ax0.axvline(f_sel, color="w", lw=1.5, alpha=0.9)
    ax0.axhline(a_sel, color="w", lw=1.5, alpha=0.9)

    fig3 = _plot_fit_and_residuals(freqs_mhz=freqs_mhz, amps_mV=amps_mV, fit=fit)
    try:  # best-effort, backend-dependent
        fig3.canvas.manager.set_window_title("DE_RF_calibration fit + residuals")
    except Exception:
        pass

    de_used = np.asarray(fit["de_used"], dtype=float)
    resid = de_used - de_fit
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(13, 4), sharex=False, sharey=False)
    _plot_residual_slices(
        ax4,
        ax5,
        freqs_mhz=freqs_mhz,
        amps_mV=amps_mV,
        resid=resid,
        f_sel_mhz=f_sel,
        a_sel_mV=a_sel,
    )
    fig4.suptitle(f"{path.name} (residual slices at f={f_sel:.2f} MHz, a={a_sel:.2f} mV)")
    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
