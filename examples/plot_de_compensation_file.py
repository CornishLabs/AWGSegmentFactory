"""
Example: plot a "DE compensation" calibration file (JSON).

This example is designed for calibration files like:
  `examples/calibrations/814_H_calFile_17.02.2022_0=0.txt`

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
      DE(freq, amp) ≈ g(freq) * sin^2((π/2) * a / v0(freq))
    where `g(freq)` is a 6th-order polynomial (in normalized frequency) and `v0(freq)` is a
    frequency-dependent knee (also a 6th-order polynomial, enforced positive).

Figure 4:
  - Residual slices (data - fit) at the same centre (freq, amp) points as Figure 2.

Usage:
  python examples/plot_de_compensation_file.py
  python examples/plot_de_compensation_file.py path/to/calFile.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from awgsegmentfactory.debug.optical_power_calibration import plot_sin2_fit_surfaces
from awgsegmentfactory.optical_power_calibration_fit import OpticalPowerCalCurve, fit_sin2_poly_model


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
        default="examples/calibrations/814_H_calFile_17.02.2022_0=0.txt",
        help="Path to a DE compensation calibration JSON file.",
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

    freqs_mhz, amps_mV, de_raw = _de_rf_grid(de_rf)
    de_used = np.clip(de_raw, 0.0, None)

    freqs_hz = freqs_mhz * 1e6
    curves = tuple(
        OpticalPowerCalCurve(
            freq_hz=float(freqs_hz[i]),
            rf_amps_mV=np.asarray(amps_mV, dtype=float),
            optical_powers=np.asarray(de_used[i, :], dtype=float),
        )
        for i in range(int(freqs_hz.size))
    )
    fit = fit_sin2_poly_model(curves, degree_g=6, degree_v0=6)
    de_fit = np.asarray(fit.predict(freqs_hz[:, None], amps_mV[None, :]), dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)
    _plot_de_rf(ax0, freqs_mhz=freqs_mhz, amps_mV=amps_mV, de=de_raw)
    _plot_power_cal(ax1, power_cal)
    fig.suptitle(path.name)
    fig.tight_layout()

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 4), sharex=False, sharey=False)
    f_sel, a_sel = _plot_de_rf_slices(
        ax2, ax3, freqs_mhz=freqs_mhz, amps_mV=amps_mV, de=de_used, de_model=de_fit
    )
    fig2.suptitle(f"{path.name} (slices at centre of ranges: f={f_sel:.2f} MHz, a={a_sel:.2f} mV)")
    fig2.tight_layout()

    # Mark slice positions on the DE heatmap (figure 1, left axis).
    ax0.axvline(f_sel, color="w", lw=1.5, alpha=0.9)
    ax0.axhline(a_sel, color="w", lw=1.5, alpha=0.9)

    fig3, _ = plot_sin2_fit_surfaces(curves, fit, title=f"{path.name} (sin2 fit)", freq_unit="MHz")
    try:  # best-effort, backend-dependent
        fig3.canvas.manager.set_window_title("DE_RF_calibration fit + residuals")
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
    )
    fig4.suptitle(f"{path.name} (residual slices at f={f_sel:.2f} MHz, a={a_sel:.2f} mV)")
    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
