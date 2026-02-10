"""Plotting helpers for optical-power calibration fits (optional; requires matplotlib)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    Sin2PolyFitResult,
    regular_grid_from_curves,
)


def _freq_scale(freq_unit: str) -> float:
    if freq_unit == "Hz":
        return 1.0
    if freq_unit == "kHz":
        return 1e3
    if freq_unit == "MHz":
        return 1e6
    raise ValueError("freq_unit must be one of {'Hz','kHz','MHz'}")


def plot_sin2_fit_surfaces(
    curves: Sequence[OpticalPowerCalCurve],
    fit: Sin2PolyFitResult,
    *,
    title: str = "Optical-power calibration fit (sin2)",
    freq_unit: str = "MHz",
) -> tuple[object, tuple[object, object, object]]:
    """
    Plot data/model/residual as 2D surfaces.

    - Uses a heatmap if data is a clean (freq,amp) grid.
    - Falls back to a 2D scatter plot for irregular sampling.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plot_sin2_fit_surfaces requires matplotlib. Install the `dev` dependency group."
        ) from exc

    freq_scale_hz = _freq_scale(freq_unit)
    xlabel = f"RF frequency ({freq_unit})"
    ylabel = "RF amplitude (mV)"

    grid = regular_grid_from_curves(curves)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharex=False, sharey=False)
    ax0, ax1, ax2 = axs

    if grid is not None:
        freqs_hz, amps_mV, p = grid
        F = freqs_hz[:, None]
        A = amps_mV[None, :]
        p_fit = fit.predict(F, A)
        resid = p - p_fit

        vmin = float(np.nanmin(p))
        vmax = float(np.nanmax(p))
        rmax = float(np.nanmax(np.abs(resid)))

        im0 = ax0.pcolormesh(
            freqs_hz / freq_scale_hz,
            amps_mV,
            p.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax0.set_title("Data")
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        plt.colorbar(im0, ax=ax0, label="Optical power (arb)")

        im1 = ax1.pcolormesh(
            freqs_hz / freq_scale_hz,
            amps_mV,
            p_fit.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax1.set_title("Model")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        plt.colorbar(im1, ax=ax1, label="Optical power (arb)")

        im2 = ax2.pcolormesh(
            freqs_hz / freq_scale_hz,
            amps_mV,
            resid.T,
            shading="auto",
            cmap="coolwarm",
            vmin=-rmax,
            vmax=rmax,
        )
        ax2.set_title("Residual (data - model)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        plt.colorbar(im2, ax=ax2, label="Residual (arb)")
    else:
        # Scatter fallback.
        ff = np.concatenate([np.full_like(c.rf_amps_mV, float(c.freq_hz), dtype=float) for c in curves], axis=0)
        aa = np.concatenate([np.asarray(c.rf_amps_mV, dtype=float).reshape(-1) for c in curves], axis=0)
        pp = np.concatenate([np.asarray(c.optical_powers, dtype=float).reshape(-1) for c in curves], axis=0)
        pp_fit = fit.predict(ff, aa)
        resid = pp - pp_fit
        rmax = float(np.nanmax(np.abs(resid)))

        sc0 = ax0.scatter(ff / freq_scale_hz, aa, c=pp, s=8, cmap="viridis")
        ax0.set_title("Data")
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        plt.colorbar(sc0, ax=ax0, label="Optical power (arb)")

        sc1 = ax1.scatter(ff / freq_scale_hz, aa, c=pp_fit, s=8, cmap="viridis")
        ax1.set_title("Model")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        plt.colorbar(sc1, ax=ax1, label="Optical power (arb)")

        sc2 = ax2.scatter(ff / freq_scale_hz, aa, c=resid, s=8, cmap="coolwarm", vmin=-rmax, vmax=rmax)
        ax2.set_title("Residual (data - model)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        plt.colorbar(sc2, ax=ax2, label="Residual (arb)")

    fig.suptitle(f"{title}   RMSE={fit.rmse:.4g}  max|res|={fit.max_abs_resid:.4g}")
    fig.tight_layout()
    return fig, (ax0, ax1, ax2)


def plot_sin2_fit_parameters(
    fit: Sin2PolyFitResult,
    *,
    title: str = "Fitted parameters vs frequency",
    freq_unit: str = "MHz",
) -> tuple[object, tuple[object, object]]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plot_sin2_fit_parameters requires matplotlib. Install the `dev` dependency group."
        ) from exc

    freq_scale_hz = _freq_scale(freq_unit)
    xlabel = f"RF frequency ({freq_unit})"

    freqs = np.asarray(fit.freqs_hz, dtype=float).reshape(-1)
    g_fit = np.asarray(fit.g_fit_by_freq, dtype=float).reshape(-1)
    v0 = np.asarray(fit.v0_mV_by_freq, dtype=float).reshape(-1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 3.8), sharex=False)
    ax0.plot(freqs / freq_scale_hz, g_fit, "C0.-", lw=1.5, ms=4)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("g(freq) (arb)")
    ax0.set_title("g(freq)")
    ax0.grid(True, alpha=0.25)

    ax1.plot(freqs / freq_scale_hz, v0, "C1.-", lw=1.5, ms=4)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("v0(freq) (mV)")
    ax1.set_title("v0(freq)")
    ax1.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_sin2_fit_slices(
    curves: Sequence[OpticalPowerCalCurve],
    fit: Sin2PolyFitResult,
    *,
    n_slices: int = 5,
    title: str = "Slices: optical power vs RF amplitude",
    freq_unit: str = "MHz",
) -> tuple[object, object]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plot_sin2_fit_slices requires matplotlib. Install the `dev` dependency group."
        ) from exc

    if not curves:
        raise ValueError("curves is empty")
    n_slices = int(n_slices)
    if n_slices <= 0:
        raise ValueError("n_slices must be > 0")

    freq_scale_hz = _freq_scale(freq_unit)

    curves_sorted = sorted(curves, key=lambda c: float(c.freq_hz))
    idxs = np.unique(np.linspace(0, len(curves_sorted) - 1, n_slices, dtype=int))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for idx in idxs:
        c = curves_sorted[int(idx)]
        a = np.asarray(c.rf_amps_mV, dtype=float).reshape(-1)
        p = np.asarray(c.optical_powers, dtype=float).reshape(-1)
        pred = fit.predict(np.full_like(a, float(c.freq_hz)), a)
        f_label = float(c.freq_hz) / freq_scale_hz
        ax.plot(a, p, ".", ms=4, alpha=0.6, label=f"{f_label:.2f} {freq_unit} data")
        ax.plot(a, pred, "-", lw=1.8, alpha=0.9, label=f"{f_label:.2f} {freq_unit} model")

    ax.set_xlabel("RF amplitude (mV)")
    ax.set_ylabel("Optical power (arb)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_sin2_fit_report(
    curves: Sequence[OpticalPowerCalCurve],
    fit: Sin2PolyFitResult,
    *,
    title: str = "Optical-power calibration fit (sin2)",
    freq_unit: str = "MHz",
) -> tuple[object, object, object]:
    """Convenience: surfaces + parameter traces + slices."""
    fig0, _ = plot_sin2_fit_surfaces(curves, fit, title=title, freq_unit=freq_unit)
    fig1, _ = plot_sin2_fit_parameters(fit, freq_unit=freq_unit)
    fig2, _ = plot_sin2_fit_slices(curves, fit, freq_unit=freq_unit)
    return fig0, fig1, fig2


def plot_sin2_fit_report_by_logical_channel(
    curves_by_logical_channel: Mapping[str, Sequence[OpticalPowerCalCurve]],
    fits_by_logical_channel: Mapping[str, Sin2PolyFitResult],
    *,
    title: str = "Optical-power calibration fits (sin2)",
    freq_unit: str = "MHz",
) -> dict[str, tuple[object, object, object]]:
    """Generate a report per logical channel."""
    out: dict[str, tuple[object, object, object]] = {}
    for lc, curves in curves_by_logical_channel.items():
        if lc not in fits_by_logical_channel:
            raise KeyError(f"Missing fit for logical_channel {lc!r}")
        out[str(lc)] = plot_sin2_fit_report(
            curves,
            fits_by_logical_channel[lc],
            title=f"{title} [{lc}]",
            freq_unit=freq_unit,
        )
    return out
