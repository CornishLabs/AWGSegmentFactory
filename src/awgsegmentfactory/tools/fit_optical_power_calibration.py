"""
Optical-power calibration tool (fit + optional diagnostics + calibration output).

Supported input formats:
- DE-compensation JSON (typically `*.txt`) with keys:
  - `DE_RF_calibration`
  - `Power_calibration` (optional metadata; overview power map is derived from `DE_RF_calibration`)
- `.awgde` JSON files (iso-power point clouds)
- `.csv` point clouds with columns:
  - `freq_mhz` / `frequency_mhz` / `freq` (MHz)
  - `rf_amp_mv` / `rf_amplitude_mv` / `amp_mv`
  - `power` / `optical_power` / `diffraction_efficiency` (arb)

Outputs:
- Fit metrics
- `AODSin2Calib` Python snippet
- Optional persisted AWG physical-setup JSON via `--write-out`
- Optional debug plots (`--plot`):
  - Input-data overview (2D DE map + DE-derived required RF-amplitude map)
  - Data/Model/Residual 2D fit surfaces with slice markers
  - Slice comparisons (data vs fit at 15%/50%/85% axis ranges)
  - Fitted parameter traces: `g(freq)` and `v0(freq)`

Usage:
  python -m awgsegmentfactory.tools.fit_optical_power_calibration
  python -m awgsegmentfactory.tools.fit_optical_power_calibration --input-data-file examples/calibrations/814_H_calFile_17.02.2022_0=0.txt --plot --write-out examples/calibrations/H_characterisation.json
  python -m awgsegmentfactory.tools.fit_optical_power_calibration --input-data-file my_scan.csv --input-data-format csv --no-plot
  python -m awgsegmentfactory.tools.fit_optical_power_calibration --input-data-file path/to/ch0.csv --input-data-file path/to/ch1.csv --logical-to-hardware-map H=0 --logical-to-hardware-map V=1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from awgsegmentfactory.calibration import AWGPhysicalSetupInfo
from awgsegmentfactory.debug.optical_power_calibration import (
    plot_sin2_fit_parameters,
    plot_sin2_fit_surfaces,
)
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    Sin2PolyFitResult,
    aod_sin2_calib_to_python,
    curves_from_awgde_dict,
    curves_from_de_rf_calibration_dict,
    curves_from_point_cloud,
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


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _csv_point_cloud_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse CSV with either:
    - header columns (preferred), or
    - no header (first 3 numeric columns are used).
    """
    first_data_row: list[str] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            first_data_row = [x.strip() for x in s.split(",")]
            break
    if first_data_row is None:
        raise ValueError(f"{path}: CSV appears empty")
    has_header = True
    if len(first_data_row) >= 3:
        try:
            float(first_data_row[0])
            float(first_data_row[1])
            float(first_data_row[2])
            has_header = False
        except Exception:
            has_header = True

    if has_header:
        arr_named = np.genfromtxt(path, delimiter=",", names=True, dtype=float, comments="#")
        names = tuple(arr_named.dtype.names or ())
        if not names:
            raise ValueError(f"{path}: CSV header parsing failed")
        norm_to_name = {_normalize_col_name(n): n for n in names}

        def pick(candidates: Sequence[str]) -> np.ndarray | None:
            for key in candidates:
                src = norm_to_name.get(key)
                if src is not None:
                    return np.asarray(arr_named[src], dtype=float).reshape(-1)
            return None

        f = pick(("freqmhz", "frequencymhz", "freq", "frequency", "f"))
        a = pick(("rfampmv", "rfamplitudemv", "ampmv", "rfamp", "amp"))
        p = pick(("opticalpower", "power", "diffractionefficiency", "de"))
        if f is None or a is None or p is None:
            if len(names) < 3:
                raise ValueError(
                    f"{path}: CSV needs either recognized headers or at least 3 numeric columns"
                )
            f = np.asarray(arr_named[names[0]], dtype=float).reshape(-1)
            a = np.asarray(arr_named[names[1]], dtype=float).reshape(-1)
            p = np.asarray(arr_named[names[2]], dtype=float).reshape(-1)
        return f, a, p

    arr = np.genfromtxt(path, delimiter=",", dtype=float, comments="#")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size < 3:
            raise ValueError(f"{path}: CSV must contain at least 3 columns")
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: CSV must contain at least 3 columns")
    return (
        np.asarray(arr[:, 0], dtype=float).reshape(-1),
        np.asarray(arr[:, 1], dtype=float).reshape(-1),
        np.asarray(arr[:, 2], dtype=float).reshape(-1),
    )


def _looks_like_awgde_payload(cal: dict) -> bool:
    if not isinstance(cal, dict) or not cal:
        return False
    for v in cal.values():
        if not isinstance(v, dict):
            continue
        if "Frequency (MHz)" in v and "RF Amplitude (mV)" in v:
            return True
    return False


def _infer_input_format(path: Path, explicit: str) -> str:
    fmt = str(explicit).strip().lower()
    if fmt and fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".awgde":
        return "awgde"
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "de-json"
    if suffix == ".json":
        cal = _load_json(path)
        if isinstance(cal, dict) and "DE_RF_calibration" in cal:
            return "de-json"
        if isinstance(cal, dict) and _looks_like_awgde_payload(cal):
            return "awgde"
        raise ValueError(f"{path}: could not infer JSON calibration format; pass --input-data-format")
    raise ValueError(f"{path}: unsupported extension; pass --input-data-format")


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


def _plot_power_map_derived_from_de_rf(
    ax,
    *,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de: np.ndarray,
) -> None:
    """
    Plot RF frequency × target optical power -> required RF amplitude,
    derived directly from the DE_RF_calibration grid used for fitting.
    """
    if de.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError(
            f"DE grid shape mismatch: de={de.shape}, freqs={freqs_mhz.shape}, amps={amps_mV.shape}"
        )

    ff_chunks: list[np.ndarray] = []
    pp_chunks: list[np.ndarray] = []
    aa_chunks: list[np.ndarray] = []

    # Invert each per-frequency DE(amp) curve into amp(DE), after monotonic cleanup.
    n_query = max(64, min(400, int(2 * max(amps_mV.size, 1))))
    for i, f_mhz in enumerate(np.asarray(freqs_mhz, dtype=float).reshape(-1)):
        a = np.asarray(amps_mV, dtype=float).reshape(-1)
        p = np.asarray(de[i, :], dtype=float).reshape(-1)
        m = np.isfinite(a) & np.isfinite(p)
        a = a[m]
        p = p[m]
        if a.size < 2:
            continue

        order = np.argsort(a, kind="stable")
        a = a[order]
        p = p[order]

        # Enforce non-decreasing DE with amplitude to get a stable inverse.
        p = np.maximum.accumulate(p)

        p_uniq, first_idx = np.unique(p, return_index=True)
        a_uniq = a[first_idx]
        if p_uniq.size < 2:
            continue
        p_lo = float(p_uniq[0])
        p_hi = float(p_uniq[-1])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            continue

        p_q = np.linspace(p_lo, p_hi, num=n_query, dtype=float)
        a_q = np.interp(p_q, p_uniq, a_uniq)

        ff_chunks.append(np.full_like(p_q, float(f_mhz), dtype=float))
        pp_chunks.append(p_q)
        aa_chunks.append(a_q)

    if not ff_chunks:
        raise ValueError("Could not derive a power->RF map from DE_RF_calibration data")

    f_all = np.concatenate(ff_chunks, axis=0)
    p_all = np.concatenate(pp_chunks, axis=0)
    a_all = np.concatenate(aa_chunks, axis=0)

    tcf = ax.tricontourf(f_all, p_all, a_all, levels=60, cmap="viridis")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("Target optical power / DE (arb)")
    ax.set_title("Derived from DE_RF_calibration: required RF amplitude")

    p_min = float(np.nanmin(p_all))
    p_max = float(np.nanmax(p_all))
    if np.isfinite(p_min) and np.isfinite(p_max):
        if p_max > p_min:
            pad = 0.03 * (p_max - p_min)
            ax.set_ylim(p_min - pad, p_max + pad)
        else:
            pad = 0.05 * max(abs(p_max), 1.0)
            ax.set_ylim(p_min - pad, p_max + pad)

    import matplotlib.pyplot as plt

    plt.colorbar(tcf, ax=ax, label="RF amplitude (mV)")


def _regular_grid_from_curves(
    curves: Sequence[OpticalPowerCalCurve],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not curves:
        return None

    curves_sorted = sorted(curves, key=lambda c: float(c.freq_hz))
    amps_mV = np.asarray(curves_sorted[0].rf_amps_mV, dtype=float).reshape(-1)
    if amps_mV.size == 0:
        return None

    n_amp = int(amps_mV.size)
    freqs_mhz = np.empty((len(curves_sorted),), dtype=float)
    de = np.empty((len(curves_sorted), n_amp), dtype=float)
    for i, c in enumerate(curves_sorted):
        a = np.asarray(c.rf_amps_mV, dtype=float).reshape(-1)
        if a.size != n_amp:
            return None
        if not np.allclose(a, amps_mV, rtol=0.0, atol=1e-9):
            return None
        freqs_mhz[i] = float(c.freq_hz) * 1e-6
        de[i, :] = np.asarray(c.optical_powers, dtype=float).reshape(-1)
    return freqs_mhz, amps_mV, de


def _slice_specs_for_axis(n_points: int) -> tuple[tuple[float, str], ...]:
    n = int(n_points)
    if n >= 3:
        return ((0.15, "15%"), (0.50, "50%"), (0.85, "85%"))
    if n == 2:
        return ((0.15, "15%"), (0.85, "85%"))
    if n == 1:
        return ((0.50, "50%"),)
    return tuple()


def _select_slice_indices(axis_values: np.ndarray) -> list[tuple[int, str]]:
    vals = np.asarray(axis_values, dtype=float).reshape(-1)
    if vals.size == 0:
        return []

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    specs = _slice_specs_for_axis(int(vals.size))

    used: set[int] = set()
    out: list[tuple[int, str]] = []
    for frac, pct_label in specs:
        target = lo + float(frac) * span
        order = np.argsort(np.abs(vals - target), kind="stable")
        pick: int | None = None
        for idx in order.tolist():
            ii = int(idx)
            if ii not in used:
                pick = ii
                break
        if pick is None:
            pick = int(order[0])
        used.add(int(pick))
        out.append((int(pick), str(pct_label)))
    return out


def _add_slice_markers(
    ax,
    *,
    freq_slices_mhz: Sequence[float],
    amp_slices_mV: Sequence[float],
) -> None:
    for f_mhz in freq_slices_mhz:
        ax.axvline(float(f_mhz), color="w", lw=0.9, alpha=0.35, zorder=2)
    for a_mV in amp_slices_mV:
        ax.axhline(float(a_mV), color="w", lw=0.9, ls="--", alpha=0.30, zorder=2)


def _plot_slice_comparison_figure(
    path: Path,
    *,
    logical_channel: str,
    freqs_mhz: np.ndarray,
    amps_mV: np.ndarray,
    de_data: np.ndarray,
    de_model: np.ndarray,
    warn_mV: float,
    critical_mV: float,
) -> tuple[list[float], list[float]]:
    if de_data.shape != de_model.shape:
        raise ValueError(
            f"Slice-plot grid shape mismatch: data={de_data.shape}, model={de_model.shape}"
        )
    if de_data.shape != (freqs_mhz.size, amps_mV.size):
        raise ValueError(
            f"Slice-plot axes mismatch: data={de_data.shape}, freqs={freqs_mhz.shape}, amps={amps_mV.shape}"
        )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example's plotting mode requires matplotlib. Install the `dev` dependency group."
        ) from exc

    freq_slice_idxs = _select_slice_indices(freqs_mhz)
    amp_slice_idxs = _select_slice_indices(amps_mV)

    fig, (ax_amp, ax_freq) = plt.subplots(
        1, 2, figsize=(14, 4.4), sharex=False, sharey=False
    )

    for i, (idx, pct) in enumerate(freq_slice_idxs):
        color = f"C{i}"
        f_sel = float(freqs_mhz[int(idx)])
        ax_amp.plot(
            amps_mV,
            de_data[int(idx), :],
            color=color,
            lw=1.6,
            label=f"data {pct} ({f_sel:.2f} MHz)",
        )
        ax_amp.plot(
            amps_mV,
            de_model[int(idx), :],
            color=color,
            lw=1.6,
            ls="--",
            label=f"fit  {pct} ({f_sel:.2f} MHz)",
        )
    ax_amp.set_xlabel("RF amplitude (mV)")
    ax_amp.set_ylabel("Optical power / DE (arb)")
    ax_amp.set_title("Frequency slices: DE vs RF amplitude")
    ax_amp.grid(True, alpha=0.25)
    ax_amp.legend(loc="best", fontsize=8, ncol=2)
    _add_rf_amp_limit_regions(
        ax_amp, amp_axis="x", warn_mV=warn_mV, critical_mV=critical_mV
    )

    for i, (idx, pct) in enumerate(amp_slice_idxs):
        color = f"C{i}"
        a_sel = float(amps_mV[int(idx)])
        ax_freq.plot(
            freqs_mhz,
            de_data[:, int(idx)],
            color=color,
            lw=1.6,
            label=f"data {pct} ({a_sel:.1f} mV)",
        )
        ax_freq.plot(
            freqs_mhz,
            de_model[:, int(idx)],
            color=color,
            lw=1.6,
            ls="--",
            label=f"fit  {pct} ({a_sel:.1f} mV)",
        )
    ax_freq.set_xlabel("RF frequency (MHz)")
    ax_freq.set_ylabel("Optical power / DE (arb)")
    ax_freq.set_title("Amplitude slices: DE vs RF frequency")
    ax_freq.grid(True, alpha=0.25)
    ax_freq.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(f"{path.name} [{logical_channel}] slices")
    fig.tight_layout()

    freq_slices_mhz = [float(freqs_mhz[int(idx)]) for idx, _ in freq_slice_idxs]
    amp_slices_mV = [float(amps_mV[int(idx)]) for idx, _ in amp_slice_idxs]
    return freq_slices_mhz, amp_slices_mV


def _plot_channel_debug_figures(
    path: Path,
    *,
    logical_channel: str,
    curves: Sequence[OpticalPowerCalCurve],
    fit: Sin2PolyFitResult,
    de_rf: dict | None,
    warn_mV: float,
    critical_mV: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example's plotting mode requires matplotlib. Install the `dev` dependency group."
        ) from exc

    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if isinstance(de_rf, dict):
        grid = _de_rf_grid(de_rf)
    if grid is None:
        grid = _regular_grid_from_curves(curves)

    freq_slices_mhz: list[float] = []
    amp_slices_mV: list[float] = []

    if grid is not None:
        freqs_mhz, amps_mV, de_raw = grid
        de_used = np.clip(np.asarray(de_raw, dtype=float), 0.0, None)
        de_fit = np.asarray(
            fit.predict((freqs_mhz * 1e6)[:, None], amps_mV[None, :]),
            dtype=float,
        )

        fig_input, (ax_in0, ax_in1) = plt.subplots(
            1, 2, figsize=(13, 5), sharex=False, sharey=False
        )
        _plot_de_rf(
            ax_in0,
            freqs_mhz=freqs_mhz,
            amps_mV=amps_mV,
            de=de_raw,
            warn_mV=warn_mV,
            critical_mV=critical_mV,
        )
        _plot_power_map_derived_from_de_rf(
            ax_in1,
            freqs_mhz=freqs_mhz,
            amps_mV=amps_mV,
            de=de_used,
        )
        fig_input.suptitle(f"{path.name} [{logical_channel}] input data")
        fig_input.tight_layout()

        freq_slices_mhz, amp_slices_mV = _plot_slice_comparison_figure(
            path,
            logical_channel=logical_channel,
            freqs_mhz=freqs_mhz,
            amps_mV=amps_mV,
            de_data=de_used,
            de_model=de_fit,
            warn_mV=warn_mV,
            critical_mV=critical_mV,
        )
    else:
        print(
            f"[{logical_channel}] input/slice plots skipped: data is not a regular "
            "(frequency x RF-amplitude) grid."
        )

    fig_surfaces, (ax_data, ax_model, ax_resid) = plot_sin2_fit_surfaces(
        curves,
        fit,
        title=f"{path.name} [{logical_channel}] (sin2 fit)",
        freq_unit="MHz",
    )
    for ax in (ax_data, ax_model, ax_resid):
        _add_rf_amp_limit_regions(
            ax, amp_axis="y", warn_mV=warn_mV, critical_mV=critical_mV
        )
        _add_slice_markers(
            ax,
            freq_slices_mhz=freq_slices_mhz,
            amp_slices_mV=amp_slices_mV,
        )
    try:  # backend-dependent
        fig_surfaces.canvas.manager.set_window_title(f"Fit surfaces [{logical_channel}]")
    except Exception:
        pass

    fig_params, _ = plot_sin2_fit_parameters(
        fit,
        title=f"{path.name} [{logical_channel}] fitted parameters",
        freq_unit="MHz",
    )
    try:  # backend-dependent
        fig_params.canvas.manager.set_window_title(f"Fit parameters [{logical_channel}]")
    except Exception:
        pass


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


def _load_curves_from_de_json_file(
    path: Path,
) -> tuple[tuple[OpticalPowerCalCurve, ...], tuple[Path, dict]]:
    cal = _load_json(path)
    de_rf = cal.get("DE_RF_calibration")
    if not isinstance(de_rf, dict):
        raise ValueError(f"{path}: expected JSON key 'DE_RF_calibration' to be a dict")
    curves = curves_from_de_rf_calibration_dict(de_rf)
    if not curves:
        raise ValueError(f"{path}: DE_RF_calibration produced no usable curves")
    payload = (path, de_rf)
    return curves, payload


def _freq_unit_scale_hz(unit: str) -> float:
    u = str(unit).strip()
    if u == "Hz":
        return 1.0
    if u == "kHz":
        return 1e3
    if u == "MHz":
        return 1e6
    raise ValueError("csv_freq_unit must be one of {'Hz', 'kHz', 'MHz'}")


def _load_curves_from_csv_file(
    path: Path,
    *,
    csv_freq_unit: str,
    freq_round_mhz: float,
    min_points_per_curve: int,
    max_points_per_curve: int | None,
) -> tuple[OpticalPowerCalCurve, ...]:
    freq_raw, amps_mV, power = _csv_point_cloud_columns(path)
    freq_hz = np.asarray(freq_raw, dtype=float) * float(_freq_unit_scale_hz(csv_freq_unit))
    curves = curves_from_point_cloud(
        freqs_hz=freq_hz,
        rf_amps_mV=np.asarray(amps_mV, dtype=float),
        optical_powers=np.asarray(power, dtype=float),
        freq_round_hz=float(freq_round_mhz) * 1e6,
        min_points_per_curve=int(min_points_per_curve),
        max_points_per_curve=max_points_per_curve,
        clamp_power_nonnegative=True,
    )
    if not curves:
        raise ValueError(f"{path}: CSV produced no usable curves")
    return curves


def _load_curves_from_input_file(
    path: Path,
    *,
    input_format: str,
    csv_freq_unit: str,
    freq_round_mhz: float,
    max_power_levels: int | None,
    min_points_per_curve: int,
    max_points_per_curve: int | None,
) -> tuple[tuple[OpticalPowerCalCurve, ...], tuple[Path, dict] | None, str]:
    fmt = _infer_input_format(path, input_format)
    if fmt == "de-json":
        curves, payload = _load_curves_from_de_json_file(path)
        return curves, payload, fmt
    if fmt == "awgde":
        curves = _load_curves_from_awgde_file(
            path,
            freq_round_mhz=freq_round_mhz,
            max_power_levels=max_power_levels,
            min_points_per_curve=min_points_per_curve,
            max_points_per_curve=max_points_per_curve,
        )
        return curves, None, fmt
    if fmt == "csv":
        curves = _load_curves_from_csv_file(
            path,
            csv_freq_unit=csv_freq_unit,
            freq_round_mhz=freq_round_mhz,
            min_points_per_curve=min_points_per_curve,
            max_points_per_curve=max_points_per_curve,
        )
        return curves, None, fmt
    raise ValueError(f"{path}: unsupported input format {fmt!r}")


def _default_input_path() -> Path:
    p = Path("examples/calibrations/814_H_calFile_17.02.2022_0=0.txt")
    if p.exists():
        return p
    p_alt = Path("examples/814_H_calFile_17.02.2022_0=0.txt")
    return p_alt


def _resolve_input_files(items: Sequence[str]) -> list[Path]:
    if not items:
        return [_default_input_path()]
    return [Path(str(x)) for x in items]


def _parse_logical_to_hardware_map(
    items: Sequence[str], *, n_ch: int
) -> dict[str, int]:
    if n_ch <= 0:
        raise ValueError("n_ch must be > 0")
    if not items:
        return {f"ch{i}": i for i in range(n_ch)}

    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"--logical-to-hardware-map must be logical=index, got {item!r}"
            )
        logical, idx_str = item.split("=", 1)
        key = str(logical).strip()
        idx = int(idx_str)
        if key in out:
            raise SystemExit(f"Duplicate logical key in --logical-to-hardware-map: {key!r}")
        out[key] = idx

    used: set[int] = set()
    for logical, idx in out.items():
        if idx < 0 or idx >= int(n_ch):
            raise SystemExit(
                f"--logical-to-hardware-map index out of range: {logical!r}={idx}, "
                f"expected 0..{int(n_ch) - 1}"
            )
        if idx in used:
            raise SystemExit(
                "--logical-to-hardware-map must be one-to-one "
                f"(duplicate hardware index {idx})"
            )
        used.add(idx)
    return out


def _resolve_traceability_strings(
    *,
    input_paths: Sequence[Path],
    traceability_flags: Sequence[str],
) -> list[str]:
    n = len(input_paths)
    provided = [str(x) for x in traceability_flags]
    if not provided:
        return [p.as_posix() for p in input_paths]
    if len(provided) == 1 and n == 1:
        return [provided[0]]
    if len(provided) != n:
        raise SystemExit(
            "--traceability-string must be provided either once for single-channel input "
            "or once per --input-data-file."
        )
    return provided


def _logical_labels_for_hw_index(
    logical_to_hardware_map: dict[str, int], *, hw_index: int
) -> str:
    labels = sorted(
        logical for logical, idx in logical_to_hardware_map.items() if int(idx) == int(hw_index)
    )
    if not labels:
        return f"ch{int(hw_index)}"
    return "/".join(labels)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data-file",
        action="append",
        default=[],
        help=(
            "Calibration input path (repeatable). "
            "If omitted, defaults to examples/calibrations/814_H_calFile_17.02.2022_0=0.txt."
        ),
    )
    parser.add_argument(
        "--input-data-format",
        choices=("auto", "de-json", "awgde", "csv"),
        default="auto",
        help="Input data format. Defaults to extension/content inference.",
    )
    parser.add_argument(
        "--logical-to-hardware-map",
        action="append",
        default=[],
        help=(
            "Logical-to-hardware mapping as logical=index (repeatable). "
            "If omitted, defaults to ch0->0, ch1->1, ..."
        ),
    )
    parser.add_argument(
        "--traceability-string",
        action="append",
        default=[],
        help=(
            "Traceability string for each channel calibration (repeatable). "
            "If omitted, each input file path is used."
        ),
    )
    parser.add_argument(
        "--var-name",
        default="AWG_CALIB",
        help=(
            "Base Python variable name used for printed constants. "
            "Per-channel constants use suffix `_CH{index}`."
        ),
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show matplotlib debug plots.",
    )
    parser.add_argument(
        "--write-out",
        default=None,
        help="If set, write a serialized AWG physical-setup JSON to this path.",
    )
    parser.add_argument(
        "--csv-freq-unit",
        choices=("Hz", "kHz", "MHz"),
        default="MHz",
        help="Frequency unit for CSV column 1 (default: MHz).",
    )
    parser.add_argument(
        "--awgde-freq-round-mhz",
        type=float,
        default=0.5,
        help="For .awgde/.csv point-cloud inputs: frequency rounding grid in MHz before grouping into curves.",
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
        help="For .awgde/.csv inputs: drop frequency bins with fewer points than this.",
    )
    parser.add_argument(
        "--awgde-max-points-per-curve",
        type=int,
        default=300,
        help="For .awgde/.csv inputs: downsample each frequency-curve to at most this many points.",
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
    args = parser.parse_args(argv)
    warn_mV = float(args.rf_amp_warn_mv)
    critical_mV = float(args.rf_amp_critical_mv)
    plot_enabled = bool(args.plot)

    input_paths = _resolve_input_files(args.input_data_file)
    n_ch = int(len(input_paths))
    channel_keys = [f"ch{i}" for i in range(n_ch)]
    traceability_strings = _resolve_traceability_strings(
        input_paths=input_paths,
        traceability_flags=args.traceability_string,
    )

    curves_by_channel: dict[str, tuple[OpticalPowerCalCurve, ...]] = {}
    de_file_payload_by_channel: dict[str, tuple[Path, dict]] = {}
    source_files_by_channel: dict[str, Path] = {}
    for idx, path in enumerate(input_paths):
        key = channel_keys[idx]
        curves, payload, _detected_format = _load_curves_from_input_file(
            path,
            input_format=str(args.input_data_format),
            csv_freq_unit=str(args.csv_freq_unit),
            freq_round_mhz=float(args.awgde_freq_round_mhz),
            max_power_levels=int(args.awgde_max_power_levels)
            if args.awgde_max_power_levels is not None
            else None,
            min_points_per_curve=int(args.awgde_min_points_per_curve),
            max_points_per_curve=int(args.awgde_max_points_per_curve)
            if args.awgde_max_points_per_curve is not None
            else None,
        )
        curves_by_channel[key] = curves
        source_files_by_channel[key] = path
        if payload is not None:
            de_file_payload_by_channel[key] = payload

    logical_to_hardware_map = _parse_logical_to_hardware_map(
        args.logical_to_hardware_map,
        n_ch=n_ch,
    )

    fits_by_lc, freq_mid_hz, freq_halfspan_hz = fit_sin2_poly_model_by_logical_channel(
        curves_by_channel,
        degree_g=6,
        degree_v0=6,
        shared_freq_norm=True,
    )
    amp_scale = suggest_amp_scale_from_curves(curves_by_channel)

    channel_calibrations: list = []
    for idx, key in enumerate(channel_keys):
        curves = curves_by_channel[key]
        freqs_hz = np.asarray([float(c.freq_hz) for c in curves], dtype=float)
        if freqs_hz.size == 0:
            raise ValueError(f"No frequency data for {key}")
        channel_calibrations.append(
            fits_by_lc[key].to_aod_sin2_calib(
                amp_scale=float(amp_scale),
                freq_min_hz=float(np.min(freqs_hz)),
                freq_max_hz=float(np.max(freqs_hz)),
                traceability_string=str(traceability_strings[idx]),
            )
        )

    physical_setup = AWGPhysicalSetupInfo(
        logical_to_hardware_map=logical_to_hardware_map,
        channel_calibrations=tuple(channel_calibrations),
    )

    print("--- saturation calibration fit ---")
    print("model: sin2")
    print("N_ch:", int(physical_setup.N_ch))
    print("logical_to_hardware_map:", dict(physical_setup.logical_to_hardware_map))
    print("freq_mid_hz:", freq_mid_hz)
    print("freq_halfspan_hz:", freq_halfspan_hz)
    print("amp_scale:", float(amp_scale))
    for idx, key in enumerate(channel_keys):
        fit = fits_by_lc[key]
        cal = channel_calibrations[idx]
        print(f"[ch{idx}] rmse={fit.rmse:.4g}  max|res|={fit.max_abs_resid:.4g}")
        print(
            f"  freq_min_hz={float(cal.freq_min_hz):.6g}  "
            f"freq_max_hz={float(cal.freq_max_hz):.6g}  best_freq_hz={int(cal.best_freq_hz)}"
        )
        print(f"  traceability_string={cal.traceability_string!r}")

    print("\n--- python constant ---")
    print("from awgsegmentfactory.calibration import AODSin2Calib, AWGPhysicalSetupInfo")
    channel_var_names: list[str] = []
    for idx, cal in enumerate(channel_calibrations):
        var_name = f"{str(args.var_name)}_CH{int(idx)}"
        channel_var_names.append(var_name)
        print(aod_sin2_calib_to_python(cal, var_name=var_name))
    tuple_suffix = "," if len(channel_var_names) == 1 else ""
    tuple_expr = "(" + ", ".join(channel_var_names) + tuple_suffix + ")"
    print(f"{str(args.var_name)} = AWGPhysicalSetupInfo(")
    print(f"    logical_to_hardware_map={dict(physical_setup.logical_to_hardware_map)!r},")
    print(f"    channel_calibrations={tuple_expr},")
    print(")")

    if args.write_out is not None:
        out = Path(str(args.write_out))
        physical_setup.to_file(out)
        print("\n--- physical setup file ---")
        print(out.as_posix())

    if not plot_enabled:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install the `dev` dependency group or use --no-plot."
        ) from exc

    for idx, key in enumerate(channel_keys):
        path = source_files_by_channel[key]
        payload = de_file_payload_by_channel.get(key)
        de_rf = payload[1] if payload is not None else None
        label = _logical_labels_for_hw_index(
            logical_to_hardware_map,
            hw_index=idx,
        )
        print(f"\n=== plots: {path} [{label}] ===")
        _plot_channel_debug_figures(
            path,
            logical_channel=label,
            curves=curves_by_channel[key],
            fit=fits_by_lc[key],
            de_rf=de_rf,
            warn_mV=warn_mV,
            critical_mV=critical_mV,
        )
    plt.show()


if __name__ == "__main__":
    main()
