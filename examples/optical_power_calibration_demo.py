"""
Example: first-order optical-power calibration.

This demonstrates `OpticalPowerToRFAmpCalib` via `AODDECalib`, where:
  - `amps` in the IR represent *desired optical power* (arbitrary units)
  - sample synthesis converts `(freq, optical_power)` -> RF synthesis amplitude

The example first plots a simple model of OpticalPower(freq, RF_amp) as a contour plot.
Then it compiles a 1-tone program with and without the calibration and overlays the
resulting waveforms.
"""

from __future__ import annotations

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.calibration import AODDECalib
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.synth_samples import compile_sequence_program


def _build_one_tone(
    *,
    sample_rate_hz: float,
    f_hz: float,
    optical_power: float,
    use_calibration: bool,
    calib: AODDECalib,
):
    b = AWGProgramBuilder().logical_channel("H")
    if use_calibration:
        b.with_calibration("aod_de", calib)

    b.define(
        "tone",
        logical_channel="H",
        freqs=[float(f_hz)],
        amps=[float(optical_power)],  # interpreted as optical power when calibration is attached
        phases=[np.pi / 2.0],  # sin(phase)=1 so the first sample reflects tone amplitude
    )
    b.segment("s0", mode="once", phase_mode="manual").tones("H").use_def("tone").hold(
        time=1e-6
    )
    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install the `dev` dependency group."
        ) from exc

    # A toy diffraction-efficiency model for demo purposes:
    #   de(freq) = polyval(coeffs, freq_hz / 1e6)   (freq in MHz)
    # Choose a gentle quadratic around 100 MHz.
    de_poly = (115, -296, 165, -42, 4)  # high->low, like numpy.polyval
    calib = AODDECalib(
        de_poly_by_logical_channel={"H": de_poly},
        freq_scale_hz=100e6,
        amp_scale=1.0,
        min_de=1e-6,
    )

    # ---- 1) Contour: OpticalPower(freq, RF_amp) ----
    freqs_hz = np.linspace(80e6, 120e6, 401, dtype=float)
    rf_amp = np.linspace(0.0, 1.0, 301, dtype=float)
    F_hz, A = np.meshgrid(freqs_hz, rf_amp, indexing="xy")

    de = np.polyval(np.array(de_poly, dtype=float), F_hz / float(calib.freq_scale_hz))
    de = np.maximum(de, float(calib.min_de))
    optical_power = de * (A / float(calib.amp_scale)) ** 2

    fig0, ax0 = plt.subplots(figsize=(8, 4))
    levels = 40
    cs = ax0.contourf(freqs_hz / 1e6, rf_amp, optical_power, levels=levels, cmap="viridis")
    fig0.colorbar(cs, ax=ax0, label="Optical power (arb)")
    ax0.set_xlabel("RF frequency (MHz)")
    ax0.set_ylabel("RF drive amplitude (arb)")
    ax0.set_title("Toy optical-power model: OpticalPower(freq, RF_amp)")

    # Overlay the RF amplitude needed to achieve a constant optical power target.
    p_target = 0.25
    a_needed = calib.rf_amps(
        freqs_hz, np.full_like(freqs_hz, p_target), logical_channel="H", xp=np
    )
    ax0.plot(freqs_hz / 1e6, a_needed, "w--", lw=2.0, label=f"target power {p_target:g}")
    ax0.legend(loc="upper right")
    fig0.tight_layout()

    # ---- 2) Compilation: with/without calibration ----
    fs = 625e6
    f0 = 100e6

    ir_uncal = _build_one_tone(
        sample_rate_hz=fs,
        f_hz=f0,
        optical_power=p_target,
        use_calibration=False,
        calib=calib,
    )
    ir_cal = _build_one_tone(
        sample_rate_hz=fs,
        f_hz=f0,
        optical_power=p_target,
        use_calibration=True,
        calib=calib,
    )

    q_uncal = quantize_resolved_ir(
        ir_uncal, logical_channel_to_hardware_channel={"H": 0}
    )
    q_cal = quantize_resolved_ir(
        ir_cal, logical_channel_to_hardware_channel={"H": 0}
    )

    full_scale = 20000
    compiled_uncal = compile_sequence_program(
        q_uncal, gain=1.0, clip=0.9, full_scale=full_scale
    )
    compiled_cal = compile_sequence_program(
        q_cal, gain=1.0, clip=0.9, full_scale=full_scale
    )

    y_uncal = compiled_uncal.segments[0].data_i16[0].astype(float) / float(full_scale)
    y_cal = compiled_cal.segments[0].data_i16[0].astype(float) / float(full_scale)

    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(y_uncal, lw=1.0, label="No calibration (amps treated as RF amp)")
    ax1.plot(y_cal, lw=1.0, label="With calibration (amps treated as optical power)")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Waveform (arb)")
    ax1.set_title("Compiled waveform (1 tone)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")
    fig1.tight_layout()

    print("--- calibration demo ---")
    print(f"freq = {f0/1e6:.3f} MHz, target optical power = {p_target:g}")
    print("uncalibrated RF amp =", p_target)
    print(
        "calibrated RF amp   =",
        float(calib.rf_amps(np.array([f0]), np.array([p_target]), logical_channel="H")[0]),
    )

    plt.show()


if __name__ == "__main__":
    main()

