"""
Example: first-order optical-power calibration.

This demonstrates `OpticalPowerToRFAmpCalib` via `AODSin2Calib`, where:
  - `amps` in the IR represent *desired optical power* (arbitrary units)
  - sample synthesis converts `(freq, optical_power)` -> RF synthesis amplitude

The example first plots a simple saturating model of OpticalPower(freq, RF_amp) as a contour plot.
Then it compiles a 1-tone program with and without the calibration and overlays the
resulting waveforms.
"""

from __future__ import annotations

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.calibration import AODSin2Calib
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.synth_samples import compile_sequence_program


def _build_one_tone(
    *,
    sample_rate_hz: float,
    f_hz: float,
    optical_power: float,
    use_calibration: bool,
    calib: AODSin2Calib,
):
    b = AWGProgramBuilder().logical_channel("H")
    if use_calibration:
        b.with_calibration("aod_sin2", calib)

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

    # A toy saturating model for demo purposes:
    #   optical_power(freq, rf_amp) = g(freq) * sin^2((π/2) * rf_amp / v0(freq))
    #
    # `AODSin2Calib` inverts this to map (freq, optical_power) -> rf_amp.
    #
    # Use x = (freq - mid)/halfspan as the polynomial coordinate (clamped in the calibration).
    g_poly = (-0.1, 0.0, 0.8)  # g(x) = 0.8 - 0.1 x^2  (high->low like numpy.polyval)
    v0_a_poly = (0.1, 1.3)  # v0_a(x) = 1.3 + 0.1 x  (kept > 0 over [-1,1])
    calib = AODSin2Calib(
        g_poly_by_logical_channel={"H": g_poly},
        v0_a_poly_by_logical_channel={"H": v0_a_poly},
        freq_mid_hz=100e6,
        freq_halfspan_hz=20e6,
        amp_scale=1.0,
        min_g=1e-6,
        min_v0_sq=1e-12,
        y_eps=1e-6,
    )

    # ---- 1) Contour: OpticalPower(freq, RF_amp) ----
    freqs_hz = np.linspace(80e6, 120e6, 401, dtype=float)
    rf_amp = np.linspace(0.0, 1.0, 301, dtype=float)
    F_hz, A = np.meshgrid(freqs_hz, rf_amp, indexing="xy")

    x = (F_hz - float(calib.freq_mid_hz)) / float(calib.freq_halfspan_hz)
    x = np.clip(x, -1.0, 1.0)
    g = np.polyval(np.asarray(g_poly, dtype=float), x)
    g = np.maximum(g, float(calib.min_g))
    v0_a = np.polyval(np.asarray(v0_a_poly, dtype=float), x)
    v0 = np.sqrt((v0_a * v0_a) + float(calib.min_v0_sq))
    optical_power = g * (np.sin((0.5 * np.pi) * (A / (float(calib.amp_scale) * v0))) ** 2)

    fig0, ax0 = plt.subplots(figsize=(8, 4))
    levels = 40
    cs = ax0.contourf(freqs_hz / 1e6, rf_amp, optical_power, levels=levels, cmap="viridis")
    fig0.colorbar(cs, ax=ax0, label="Optical power (arb)")
    ax0.set_xlabel("RF frequency (MHz)")
    ax0.set_ylabel("RF drive amplitude (arb)")
    ax0.set_title("Toy optical-power model: g(freq) * sin^2((π/2) * RF_amp / v0(freq))")

    # Overlay the RF amplitude needed to achieve a constant optical power target.
    p_target = 0.6
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
