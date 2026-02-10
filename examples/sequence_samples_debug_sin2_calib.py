"""
Example: sample-level debug view with `AODSin2Calib` optical-power calibration.

This demonstrates an end-to-end flow:
1) Load a DE compensation JSON file (DE vs RF amplitude vs frequency)
2) Fit a smooth, monotone saturation model:
     DE(freq, a_mV) ≈ g(freq) * sin^2((π/2) * a_mV / v0(freq))
3) Build an `AODSin2Calib` from the fitted polynomials
4) Attach it in `AWGPhysicalSetupInfo(channel_calibrations=...)` so `amps` in
   the IR represent *optical power* (or DE proxy), while sample synthesis uses
   calibrated RF amplitudes.
5) Visualize the compiled samples + segment boundaries.

In Jupyter, you may want:
  %matplotlib widget
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from awgsegmentfactory import AWGProgramBuilder, ResolvedIR
from awgsegmentfactory.calibration import AODSin2Calib, AWGPhysicalSetupInfo
from awgsegmentfactory.debug import sequence_samples_debug
from awgsegmentfactory.optical_power_calibration_fit import (
    Sin2PolyFitResult,
    curves_from_de_rf_calibration_dict,
    fit_sin2_poly_model,
    suggest_amp_scale_from_curves,
)


def _maybe_enable_matplotlib_widget_backend() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None or getattr(ip, "kernel", None) is None:
            return
        ip.run_line_magic("matplotlib", "widget")
    except Exception:
        return


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fit_calibration_file_to_sin2(
    path: Path,
) -> tuple[AODSin2Calib, Sin2PolyFitResult, float]:
    """
    Returns (calib, fit, rf_amp_max_mV).

    `rf_amp_max_mV` is useful for choosing a sensible `amp_scale` so the returned RF
    amplitudes land in a comfortable [0,1] AWG amplitude range.
    """
    cal = _load_json(path)
    de_rf = cal.get("DE_RF_calibration")
    if not isinstance(de_rf, dict):
        raise ValueError("Expected JSON key 'DE_RF_calibration' to be a dict")

    curves = curves_from_de_rf_calibration_dict(de_rf)
    if not curves:
        raise ValueError("No usable calibration curves found in DE_RF_calibration")
    fit = fit_sin2_poly_model(curves, degree_g=6, degree_v0=6)

    # Choose an amp_scale that maps ~max RF amplitude (mV) to ~1.0 (AWG units).
    amp_scale = suggest_amp_scale_from_curves(curves)
    rf_amp_max_mV = 1.0 / float(amp_scale)

    calib = fit.to_aod_sin2_calib(amp_scale=float(amp_scale))
    return calib, fit, rf_amp_max_mV


def _build_demo_program(
    *,
    sample_rate_hz: float,
    fit: Sin2PolyFitResult,
) -> ResolvedIR:
    # Single tone: hold, then linear chirp from 81 MHz -> 119 MHz.
    f_start_hz = 81e6
    f_stop_hz = 119e6
    df_hz = float(f_stop_hz - f_start_hz)

    # Choose a constant target optical power that is safely below g(freq) over the whole chirp.
    ff = np.linspace(float(f_start_hz), float(f_stop_hz), 257, dtype=float)
    g_min = float(np.min(np.maximum(fit.g_of_freq(ff), 1e-6)))
    power_frac = 0.6
    p_target = float(power_frac) * g_min

    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .define(
            "tone",
            logical_channel="H",
            freqs=[float(f_start_hz)],
            amps=[p_target],  # interpreted as optical power when calibration is attached
            phases=[np.pi / 2.0],  # sin(phase)=1 so sample[0] reflects amplitude
        )
    )

    b.segment("hold", mode="once", phase_mode="manual")
    b.tones("H").use_def("tone")
    b.hold(time=1e-6)

    # Keep optical power constant while chirping frequency. The calibration will adjust RF
    # amplitudes across the chirp as g(freq) and v0(freq) vary.
    b.segment("chirp", mode="once", phase_mode="manual")
    b.tones("H").move(df=df_hz, time=4e-6, kind="linear")

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent
            / "calibrations"
            / "814_H_calFile_17.02.2022_0=0.txt"
        ),
        help="Path to a DE compensation calibration JSON file.",
    )
    args = parser.parse_args()

    _maybe_enable_matplotlib_widget_backend()

    cal_path = Path(args.path)
    calib, fit, rf_amp_max_mV = _fit_calibration_file_to_sin2(cal_path)

    print("--- AODSin2Calib (from DE file fit) ---")
    print("file:", cal_path.name)
    print("fit rmse:", float(fit.rmse))
    print("rf_amp_max_mV:", rf_amp_max_mV, "-> amp_scale:", float(calib.amp_scale))

    # Use a realistic Spectrum-like sample rate, but keep segment durations short.
    fs = 625e6
    ir = _build_demo_program(sample_rate_hz=fs, fit=fit)
    physical_setup = AWGPhysicalSetupInfo(
        logical_to_hardware_map={"H": 0},
        channel_calibrations=(calib,),
    )

    fig, axs, slider = sequence_samples_debug(
        ir,
        physical_setup=physical_setup,
        wait_trig_loops=3,
        include_wrap_preview=True,
        window_samples=None,
        show_slider=False,
        show_markers=True,
        title="Sequence samples (AODSin2Calib compile-time calibration)",
    )

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
