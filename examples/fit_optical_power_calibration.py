"""
Combined optical-power calibration example (fit + file diagnostics).

This script combines the previous:
- `examples/fit_optical_power_calibration.py` (fit + constant snippet), and
- `examples/plot_de_compensation_file.py` (DE/Power file diagnostics).

Supported input formats:
- DE-compensation JSON (typically `*.txt`) with:
  - `DE_RF_calibration`
  - `Power_calibration` (optional for overview plot)
- `.awgde` JSON files (iso-power point clouds)

Outputs:
- Fit metrics
- `AODSin2Calib` Python snippet
- Optional plots:
  - Per-input DE/Power diagnostics (for DE-compensation JSON files)
  - Per-logical-channel fit report (data/model/residual + parameter/slice plots)

Usage:
  python examples/fit_optical_power_calibration.py
  python examples/fit_optical_power_calibration.py --input H=path/to/H_cal.txt --input V=path/to/V_cal.txt
  python examples/fit_optical_power_calibration.py --input H=path/to/file.awgde
  python examples/fit_optical_power_calibration.py --no-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from awgsegmentfactory.calibration import AODSin2Calib
from awgsegmentfactory.debug.optical_power_calibration import plot_sin2_fit_surfaces
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    Sin2PolyFitResult,
    aod_sin2_calib_to_python,
    build_aod_sin2_calib_from_fits,
    curves_from_awgde_dict,
    curves_from_de_rf_calibration_dict,
    fit_sin2_poly_model_by_logical_channel,
    suggest_amp_scale_from_curves,
)


def _add_rf_amp_limit_regions(
    ax,
    *,
    amp_axis: str,
    warn_mV: float,
    critical_mV: float,
) -> None:
    """
    Overlay RF-amplitude caution regions.

    - light hatched red:  warn_mV .. critical_mV
    - darker red:         >= critical_mV
    """
    warn = float(warn_mV)
    crit = float(critical_mV)
    if warn <= 0.0:
        return
    if crit <= warn:
        crit = warn

    if amp_axis == "y":
        lo, hi = ax.get_ylim()
        ymin, ymax = (lo, hi) if lo <= hi else (hi, lo)
        if ymax <= warn:
            return
        y_warn = max(ymin, warn)
        y_crit = max(ymin, crit)
        if y_warn < min(y_crit, ymax):
            ax.axhspan(
                y_warn,
                min(y_crit, ymax),
                facecolor="red",
                edgecolor="red",
                alpha=0.08,
                hatch="////",
                linewidth=0.0,
                zorder=1,
            )
        if y_crit < ymax:
            ax.axhspan(
                y_crit,
                ymax,
                facecolor="red",
                edgecolor="red",
                alpha=0.20,
                linewidth=0.0,
                zorder=1,
            )
        ax.set_ylim(lo, hi)
        return

    if amp_axis == "x":
        lo, hi = ax.get_xlim()
        xmin, xmax = (lo, hi) if lo <= hi else (hi, lo)
        if xmax <= warn:
            return
        x_warn = max(xmin, warn)
        x_crit = max(xmin, crit)
        if x_warn < min(x_crit, xmax):
            ax.axvspan(
                x_warn,
                min(x_crit, xmax),
                facecolor="red",
                edgecolor="red",
                alpha=0.08,
                hatch="////",
                linewidth=0.0,
                zorder=1,
            )
        if x_crit < xmax:
            ax.axvspan(
                x_crit,
                xmax,
                facecolor="red",
                edgecolor="red",
                alpha=0.20,
                linewidth=0.0,
                zorder=1,
            )
        ax.set_xlim(lo, hi)
        return

    raise ValueError("amp_axis must be 'x' or 'y'")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_de_file(cal: dict) -> None:
    keys = list(cal.keys())
    print("--- file summary ---")
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
    freq_keys = sorted(de_rf.keys(), key=lambda k: float(k))
    freqs_mhz = np.array([float(k) for k in freq_keys], dtype=float)

    amps_mV = np.asarray(de_rf[freq_keys[0]]["RF Amplitude (mV)"], dtype=float)

    de = np.empty((len(freq_keys), len(amps_mV)), dtype=float)
    for i, k in enumerate(freq_keys):
        row = np.asarray(de_rf[k]["Diffraction Efficiency"], dtype=float)
        if row.shape != amps_mV.shape:
            raise ValueError(
                f"DE_RF_calibration[{k!r}]: expected {amps_mV.shape} values, got {row.shape}"
            )
        de[i, :] = row
    return freqs_mhz, amps_mV, de


def _plot_de_rf(
    ax,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de: np.ndarray,
    warn_mV: float,
    critical_mV: float,
) -> None:
    if de.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError(
            f"DE grid shape mismatch: de={de.shape}, freqs={freqs_mhz.shape}, amps={amps_mV.shape}"
        )

    pcm = ax.pcolormesh(freqs_mhz, amps_mV, de.T, shading="auto", cmap="viridis")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("RF amplitude (mV)")
    ax.set_title("DE_RF_calibration: Diffraction efficiency")

    import matplotlib.pyplot as plt

    plt.colorbar(pcm, ax=ax, label="Diffraction efficiency (arb)")
    _add_rf_amp_limit_regions(
        ax, amp_axis="y", warn_mV=warn_mV, critical_mV=critical_mV
    )


def _plot_power_cal(ax, power_cal: dict) -> None:
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

        uniq_f, inv = np.unique(f, return_inverse=True)
        a_sum = np.zeros_like(uniq_f, dtype=float)
        counts = np.zeros_like(uniq_f, dtype=int)
        np.add.at(a_sum, inv, a)
        np.add.at(counts, inv, 1)
        a_mean = a_sum / counts

        fs.append(uniq_f)
        amps.append(a_mean)
        ps.append(np.full_like(uniq_f, p, dtype=float))

    if not fs:
        raise ValueError("Power_calibration has no finite points")

    f_all = np.concatenate(fs, axis=0)
    p_all = np.concatenate(ps, axis=0)
    a_all = np.concatenate(amps, axis=0)

    tcf = ax.tricontourf(f_all, p_all, a_all, levels=60, cmap="viridis")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("Target optical power (0..1)")
    ax.set_title("Power_calibration: required RF amplitude")

    import matplotlib.pyplot as plt

    plt.colorbar(tcf, ax=ax, label="RF amplitude (mV)")


def _plot_de_rf_slices(
    ax_amp,
    ax_freq,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de: np.ndarray,
    de_model: np.ndarray | None = None,
    warn_mV: float,
    critical_mV: float,
) -> tuple[float, float]:
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

    ax_amp.plot(amps_mV, de[i_f, :], lw=1.5, label="data")
    if de_model is not None:
        ax_amp.plot(amps_mV, de_model[i_f, :], lw=1.5, ls="--", label="model")
    ax_amp.set_xlabel("RF amplitude (mV)")
    ax_amp.set_ylabel("Diffraction efficiency (arb)")
    ax_amp.set_title(f"DE vs RF amplitude @ {f_sel:.2f} MHz")
    ax_amp.grid(True, alpha=0.25)
    if de_model is not None:
        ax_amp.legend(loc="best")
    _add_rf_amp_limit_regions(
        ax_amp, amp_axis="x", warn_mV=warn_mV, critical_mV=critical_mV
    )

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


def _plot_residual_slices(
    ax_amp,
    ax_freq,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    resid: np.ndarray,
    f_sel_mhz: float,
    a_sel_mV: float,
    warn_mV: float,
    critical_mV: float,
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
    _add_rf_amp_limit_regions(
        ax_amp, amp_axis="x", warn_mV=warn_mV, critical_mV=critical_mV
    )

    ax_freq.plot(freqs_mhz, resid[:, i_a], lw=1.5)
    ax_freq.axhline(0.0, color="k", lw=1.0, alpha=0.35)
    ax_freq.set_xlabel("RF frequency (MHz)")
    ax_freq.set_ylabel("Residual (arb)")
    ax_freq.set_title(f"Residual vs frequency @ {a_sel:.2f} mV")
    ax_freq.grid(True, alpha=0.25)


def _plot_de_file_overview(
    path: Path,
    *,
    logical_channel: str,
    fit: Sin2PolyFitResult,
    de_rf: dict,
    power_cal: dict | None,
    warn_mV: float,
    critical_mV: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example's plotting mode requires matplotlib. Install the `dev` dependency group."
        ) from exc

    freqs_mhz, amps_mV, de_raw = _de_rf_grid(de_rf)
    de_used = np.clip(de_raw, 0.0, None)
    freqs_hz = freqs_mhz * 1e6
    de_fit = np.asarray(fit.predict(freqs_hz[:, None], amps_mV[None, :]), dtype=float)

    if power_cal:
        fig1, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)
        _plot_de_rf(
            ax0,
            freqs_mhz=freqs_mhz,
            amps_mV=amps_mV,
            de=de_raw,
            warn_mV=warn_mV,
            critical_mV=critical_mV,
        )
        _plot_power_cal(ax1, power_cal)
    else:
        fig1, ax0 = plt.subplots(1, 1, figsize=(7, 5), sharex=False, sharey=False)
        _plot_de_rf(
            ax0,
            freqs_mhz=freqs_mhz,
            amps_mV=amps_mV,
            de=de_raw,
            warn_mV=warn_mV,
            critical_mV=critical_mV,
        )
    fig1.suptitle(f"{path.name} [{logical_channel}]")
    fig1.tight_layout()

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 4), sharex=False, sharey=False)
    f_sel, a_sel = _plot_de_rf_slices(
        ax2,
        ax3,
        freqs_mhz=freqs_mhz,
        amps_mV=amps_mV,
        de=de_used,
        de_model=de_fit,
        warn_mV=warn_mV,
        critical_mV=critical_mV,
    )
    fig2.suptitle(
        f"{path.name} [{logical_channel}] "
        f"(slices at f={f_sel:.2f} MHz, a={a_sel:.2f} mV)"
    )
    fig2.tight_layout()

    ax0.axvline(f_sel, color="w", lw=1.5, alpha=0.9)
    ax0.axhline(a_sel, color="w", lw=1.5, alpha=0.9)

    curves = curves_from_de_rf_calibration_dict(de_rf)
    fig3, _ = plot_sin2_fit_surfaces(
        curves,
        fit,
        title=f"{path.name} [{logical_channel}] (sin2 fit)",
        freq_unit="MHz",
    )
    for ax in fig3.axes:
        if "RF amplitude (mV)" in str(ax.get_ylabel()):
            _add_rf_amp_limit_regions(
                ax, amp_axis="y", warn_mV=warn_mV, critical_mV=critical_mV
            )
    try:  # backend-dependent
        fig3.canvas.manager.set_window_title(f"DE fit overview [{logical_channel}]")
    except Exception:
        pass

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
        warn_mV=warn_mV,
        critical_mV=critical_mV,
    )
    fig4.suptitle(f"{path.name} [{logical_channel}] residual slices")
    fig4.tight_layout()


def _load_curves_from_awgde_file(
    path: Path,
    *,
    freq_round_mhz: float,
    max_power_levels: int | None,
    min_points_per_curve: int,
    max_points_per_curve: int | None,
) -> tuple[OpticalPowerCalCurve, ...]:
    cal = _load_json(path)
    if not isinstance(cal, dict):
        raise ValueError(f"{path}: expected top-level JSON object for .awgde")
    curves = curves_from_awgde_dict(
        cal,
        freq_round_hz=float(freq_round_mhz) * 1e6,
        max_power_levels=max_power_levels,
        min_points_per_curve=int(min_points_per_curve),
        max_points_per_curve=max_points_per_curve,
    )
    if not curves:
        raise ValueError(f"{path}: .awgde produced no usable curves")
    return curves


def _default_input_path() -> Path:
    p = Path("examples/calibrations/814_H_calFile_17.02.2022_0=0.txt")
    if p.exists():
        return p
    p_alt = Path("examples/814_H_calFile_17.02.2022_0=0.txt")
    return p_alt


def _parse_inputs(items: Sequence[str]) -> list[tuple[str, Path]]:
    if not items:
        return [("H", _default_input_path())]
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--input must be logical_channel=path, got {item!r}")
        lc, p = item.split("=", 1)
        out.append((lc.strip(), Path(p)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Calibration input in the form logical_channel=path (repeatable).",
    )
    parser.add_argument(
        "--var-name",
        default="AOD_SIN2_CALIB",
        help="Python variable name used in the printed constant snippet.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots (print fit report/constants only).",
    )
    parser.add_argument(
        "--overview-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show DE-compensation diagnostics (former plot_de_compensation_file behavior) for DE JSON inputs.",
    )
    parser.add_argument(
        "--awgde-freq-round-mhz",
        type=float,
        default=0.5,
        help="For .awgde inputs: frequency rounding grid in MHz before grouping into curves.",
    )
    parser.add_argument(
        "--awgde-max-power-levels",
        type=int,
        default=300,
        help="For .awgde inputs: downsample to at most this many optical-power levels.",
    )
    parser.add_argument(
        "--awgde-min-points-per-curve",
        type=int,
        default=20,
        help="For .awgde inputs: drop frequency bins with fewer points than this.",
    )
    parser.add_argument(
        "--awgde-max-points-per-curve",
        type=int,
        default=300,
        help="For .awgde inputs: downsample each frequency-curve to at most this many points.",
    )
    parser.add_argument(
        "--rf-amp-warn-mv",
        type=float,
        default=100.0,
        help=(
            "RF amplitude warning threshold (mV): light red hatched shading starts above this value. "
            "Set <=0 to disable shading."
        ),
    )
    parser.add_argument(
        "--rf-amp-critical-mv",
        type=float,
        default=200.0,
        help="RF amplitude critical threshold (mV): darker shading starts above this value.",
    )
    args = parser.parse_args()
    warn_mV = float(args.rf_amp_warn_mv)
    critical_mV = float(args.rf_amp_critical_mv)

    inputs = _parse_inputs(args.input)

    curves_by_lc: dict[str, tuple[OpticalPowerCalCurve, ...]] = {}
    de_file_payload_by_lc: dict[str, tuple[Path, dict, dict | None]] = {}
    for lc, path in inputs:
        if path.suffix.lower() == ".awgde":
            curves_by_lc[str(lc)] = _load_curves_from_awgde_file(
                path,
                freq_round_mhz=float(args.awgde_freq_round_mhz),
                max_power_levels=int(args.awgde_max_power_levels)
                if args.awgde_max_power_levels is not None
                else None,
                min_points_per_curve=int(args.awgde_min_points_per_curve),
                max_points_per_curve=int(args.awgde_max_points_per_curve)
                if args.awgde_max_points_per_curve is not None
                else None,
            )
            continue

        cal = _load_json(path)
        de_rf = cal.get("DE_RF_calibration")
        power_cal = cal.get("Power_calibration")
        if not isinstance(de_rf, dict):
            raise ValueError(f"{path}: expected JSON key 'DE_RF_calibration' to be a dict")
        curves = curves_from_de_rf_calibration_dict(de_rf)
        if not curves:
            raise ValueError(f"{path}: DE_RF_calibration produced no usable curves")
        curves_by_lc[str(lc)] = curves
        de_file_payload_by_lc[str(lc)] = (
            path,
            de_rf,
            power_cal if isinstance(power_cal, dict) else None,
        )

    fits_by_lc, freq_mid_hz, freq_halfspan_hz = fit_sin2_poly_model_by_logical_channel(
        curves_by_lc,
        degree_g=6,
        degree_v0=6,
        shared_freq_norm=True,
    )
    amp_scale = suggest_amp_scale_from_curves(curves_by_lc)
    calib = build_aod_sin2_calib_from_fits(
        fits_by_lc,
        amp_scale=float(amp_scale),
    )
    if not isinstance(calib, AODSin2Calib):  # pragma: no cover
        raise TypeError(f"Expected AODSin2Calib, got {type(calib)!r}")

    print("--- saturation calibration fit ---")
    print("model: sin2")
    print("logical_channels:", sorted(curves_by_lc.keys()))
    print("freq_mid_hz:", freq_mid_hz)
    print("freq_halfspan_hz:", freq_halfspan_hz)
    print("amp_scale:", float(amp_scale))
    for lc, fit in fits_by_lc.items():
        print(f"[{lc}] rmse={fit.rmse:.4g}  max|res|={fit.max_abs_resid:.4g}")

    print("\n--- python constant ---")
    print("from awgsegmentfactory.calibration import AODSin2Calib")
    print(aod_sin2_calib_to_python(calib, var_name=str(args.var_name)))

    if args.no_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install the `dev` dependency group or use --no-plots."
        ) from exc

    if bool(args.overview_plots):
        for lc in sorted(curves_by_lc.keys()):
            payload = de_file_payload_by_lc.get(lc)
            if payload is None:
                print(f"[{lc}] overview plots skipped (no DE-compensation JSON payload, likely .awgde input).")
                continue
            path, de_rf, power_cal = payload
            print(f"\n=== overview: {path} [{lc}] ===")
            _summarize_de_file(_load_json(path))
            _plot_de_file_overview(
                path,
                logical_channel=lc,
                fit=fits_by_lc[lc],
                de_rf=de_rf,
                power_cal=power_cal,
                warn_mV=warn_mV,
                critical_mV=critical_mV,
            )

    from awgsegmentfactory.debug import plot_sin2_fit_report_by_logical_channel

    reports_by_lc = plot_sin2_fit_report_by_logical_channel(
        curves_by_lc, fits_by_lc, title="sin2 calibration fit"
    )
    for report in reports_by_lc.values():
        fig_surfaces, _fig_params, fig_slices = report
        for ax in fig_surfaces.axes:
            if "RF amplitude (mV)" in str(ax.get_ylabel()):
                _add_rf_amp_limit_regions(
                    ax, amp_axis="y", warn_mV=warn_mV, critical_mV=critical_mV
                )
        for ax in fig_slices.axes:
            if "RF amplitude (mV)" in str(ax.get_xlabel()):
                _add_rf_amp_limit_regions(
                    ax, amp_axis="x", warn_mV=warn_mV, critical_mV=critical_mV
                )
    plt.show()


if __name__ == "__main__":
    main()
