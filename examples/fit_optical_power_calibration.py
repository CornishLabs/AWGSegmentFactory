"""
Example: fit `AODTanh2Calib` from measured (rf_freq, rf_amp) -> optical_power data.

This script is meant as reusable "calibration tooling" for labs that need to repeat the
procedure across many AODs and many RF channels.

Currently supported input adapter:
- DE compensation JSON files like `examples/814_H_calFile_17.02.2022_0=0.txt`
  (even though the extension is `.txt`, the content is a single JSON object).

The output is:
- a printed Python constant snippet (copy/paste into your experiment code), and
- optional fit-quality plots (data/model/residual, parameter traces, and slices).

Usage:
  python examples/fit_optical_power_calibration.py
  python examples/fit_optical_power_calibration.py --input H=path/to/H_cal.txt --input V=path/to/V_cal.txt
  python examples/fit_optical_power_calibration.py --no-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awgsegmentfactory.calibration import AODTanh2Calib
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    aod_tanh2_calib_to_python,
    build_aod_tanh2_calib_from_fits,
    curves_from_de_rf_calibration_dict,
    fit_tanh2_poly_model_by_logical_channel,
    suggest_amp_scale_from_curves,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_curves_from_de_file(path: Path) -> tuple[OpticalPowerCalCurve, ...]:
    cal = _load_json(path)
    de_rf = cal.get("DE_RF_calibration")
    if not isinstance(de_rf, dict):
        raise ValueError(f"{path}: expected JSON key 'DE_RF_calibration' to be a dict")
    curves = curves_from_de_rf_calibration_dict(de_rf)
    if not curves:
        raise ValueError(f"{path}: DE_RF_calibration produced no usable curves")
    return curves


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
        default="AOD_TANH2_CALIB",
        help="Python variable name used in the printed constant snippet.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots (print constants only).",
    )
    args = parser.parse_args()

    inputs: list[tuple[str, Path]] = []
    if not args.input:
        inputs = [("H", Path("examples/814_H_calFile_17.02.2022_0=0.txt"))]
    else:
        for item in args.input:
            if "=" not in item:
                raise SystemExit(f"--input must be logical_channel=path, got {item!r}")
            lc, p = item.split("=", 1)
            inputs.append((lc.strip(), Path(p)))

    curves_by_lc: dict[str, tuple[OpticalPowerCalCurve, ...]] = {}
    for lc, path in inputs:
        curves_by_lc[str(lc)] = _load_curves_from_de_file(path)

    fits_by_lc, freq_mid_hz, freq_halfspan_hz = fit_tanh2_poly_model_by_logical_channel(
        curves_by_lc,
        degree_g=6,
        degree_v0=6,
        shared_freq_norm=True,
    )
    amp_scale = suggest_amp_scale_from_curves(curves_by_lc)
    calib: AODTanh2Calib = build_aod_tanh2_calib_from_fits(
        fits_by_lc,
        amp_scale=float(amp_scale),
    )

    print("--- tanh2 calibration fit ---")
    print("logical_channels:", sorted(curves_by_lc.keys()))
    print("freq_mid_hz:", freq_mid_hz)
    print("freq_halfspan_hz:", freq_halfspan_hz)
    print("amp_scale:", float(amp_scale))
    for lc, fit in fits_by_lc.items():
        print(f"[{lc}] rmse={fit.rmse:.4g}  max|res|={fit.max_abs_resid:.4g}")

    print("\n--- python constant ---")
    print("from awgsegmentfactory.calibration import AODTanh2Calib")
    print(aod_tanh2_calib_to_python(calib, var_name=str(args.var_name)))

    if args.no_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install the `dev` dependency group or use --no-plots."
        ) from exc

    from awgsegmentfactory.debug import plot_tanh2_fit_report_by_logical_channel

    plot_tanh2_fit_report_by_logical_channel(curves_by_lc, fits_by_lc, title="tanh2 calibration fit")
    plt.show()


if __name__ == "__main__":
    main()

