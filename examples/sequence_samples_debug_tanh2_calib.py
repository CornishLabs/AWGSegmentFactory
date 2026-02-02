"""
Example: sample-level debug view with `AODTanh2Calib` optical-power calibration.

This demonstrates an end-to-end flow:
1) Load a DE compensation JSON file (DE vs RF amplitude vs frequency)
2) Fit a smooth, monotone saturation model:
     DE(freq, a_mV) ≈ g(freq) * tanh^2(a_mV / v0(freq))
3) Build an `AODTanh2Calib` from the fitted polynomials
4) Attach it to an `AWGProgramBuilder` so `amps` in the IR represent *optical power*
   (or DE proxy), while sample synthesis uses calibrated RF amplitudes.
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
from awgsegmentfactory.calibration import AODTanh2Calib
from awgsegmentfactory.debug import sequence_samples_debug

import plot_de_compensation_file as de_fit


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


def _polyval_high_to_low(coeffs_high_to_low: tuple[float, ...], x: float) -> float:
    c = np.asarray(coeffs_high_to_low, dtype=float).reshape(-1)
    return float(np.polyval(c, float(x)))


def _fit_calibration_file_to_tanh2(path: Path) -> tuple[AODTanh2Calib, dict[str, object], float]:
    """
    Returns (calib, fit_dict, rf_amp_max_mV).

    `rf_amp_max_mV` is useful for choosing a sensible `amp_scale` so the returned RF
    amplitudes land in a comfortable [0,1] AWG amplitude range.
    """
    cal = _load_json(path)
    de_rf = cal.get("DE_RF_calibration")
    if not isinstance(de_rf, dict):
        raise ValueError("Expected JSON key 'DE_RF_calibration' to be a dict")

    freqs_mhz, amps_mV, de = de_fit._de_rf_grid(de_rf)
    fit = de_fit._fit_sat_poly_model(
        freqs_mhz=freqs_mhz,
        amps_mV=amps_mV,
        de=de,
        degree_g=6,
        degree_knee=6,
        sat_model="tanh2",
    )

    coeffs_g = tuple(float(x) for x in np.asarray(fit["coeffs_g_high_to_low"], dtype=float).tolist())
    coeffs_v0_a = tuple(float(x) for x in np.asarray(fit["coeffs_a0_high_to_low"], dtype=float).tolist())

    freq_mid_hz = float(fit["freq_mid_mhz"]) * 1e6
    freq_halfspan_hz = float(fit["freq_halfspan_mhz"]) * 1e6

    rf_amp_max_mV = float(np.max(np.asarray(amps_mV, dtype=float)))
    if not np.isfinite(rf_amp_max_mV) or rf_amp_max_mV <= 0:
        raise ValueError("Invalid RF amplitude grid in calibration file")

    # Choose an amp_scale that maps ~max RF amplitude (mV) to ~1.0 (AWG units).
    amp_scale = 1.0 / rf_amp_max_mV

    calib = AODTanh2Calib(
        g_poly_by_logical_channel={"*": coeffs_g},
        v0_a_poly_by_logical_channel={"*": coeffs_v0_a},
        freq_mid_hz=freq_mid_hz,
        freq_halfspan_hz=freq_halfspan_hz,
        amp_scale=float(amp_scale),
        # v0 is in mV, so this is in mV^2.
        min_v0_sq=float(fit["min_u0_mV2"]),
        y_eps=1e-6,
    )
    return calib, fit, rf_amp_max_mV


def _build_demo_program(*, sample_rate_hz: float, calib: AODTanh2Calib) -> ResolvedIR:
    # Single tone: hold, then linear chirp from 81 MHz -> 119 MHz.
    f_start_hz = 81e6
    f_stop_hz = 119e6
    df_hz = float(f_stop_hz - f_start_hz)

    def g_at(freq_hz: float) -> float:
        x = (float(freq_hz) - float(calib.freq_mid_hz)) / float(calib.freq_halfspan_hz)
        x = float(np.clip(x, -1.0, 1.0))
        g = _polyval_high_to_low(calib.g_poly_by_logical_channel["*"], x)
        return float(max(g, 1e-6))

    # Choose a constant target optical power that is safely below g(freq) over the whole chirp.
    ff = np.linspace(float(f_start_hz), float(f_stop_hz), 257, dtype=float)
    g_min = float(np.min([g_at(f) for f in ff]))
    power_frac = 0.6
    p_target = float(power_frac) * g_min

    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .with_calibration("aod_tanh2", calib)
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
    b.hold(time=0.5e-6)

    # Keep optical power constant while chirping frequency. The calibration will adjust RF
    # amplitudes across the chirp as g(freq) and v0(freq) vary.
    b.segment("chirp", mode="once", phase_mode="manual")
    b.tones("H").move(df=df_hz, time=2e-6, kind="linear")

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("814_H_calFile_17.02.2022_0=0.txt")),
        help="Path to a DE compensation calibration JSON file.",
    )
    args = parser.parse_args()

    _maybe_enable_matplotlib_widget_backend()

    cal_path = Path(args.path)
    calib, fit, rf_amp_max_mV = _fit_calibration_file_to_tanh2(cal_path)

    print("--- AODTanh2Calib (from DE file fit) ---")
    print("file:", cal_path.name)
    print("fit rmse:", float(fit["rmse"]))
    print("rf_amp_max_mV:", rf_amp_max_mV, "-> amp_scale:", float(calib.amp_scale))

    # Use a realistic Spectrum-like sample rate, but keep segment durations short.
    fs = 625e6
    ir = _build_demo_program(sample_rate_hz=fs, calib=calib)

    fig, axs, slider = sequence_samples_debug(
        ir,
        logical_channel_to_hardware_channel={"H": 0},
        wait_trig_loops=3,
        include_wrap_preview=True,
        window_samples=None,
        show_slider=False,
        show_markers=True,
        title="Sequence samples (AODTanh2Calib attached)",
    )

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
