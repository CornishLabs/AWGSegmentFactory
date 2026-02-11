import unittest

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.calibration import AODSin2Calib, AWGPhysicalSetupInfo
from awgsegmentfactory.debug import sequence_samples_debug


def _build_chirp_ir():
    return (
        AWGProgramBuilder()
        .logical_channel("H")
        .define(
            "tone",
            logical_channel="H",
            freqs=[80e6],
            amps=[0.25],
            phases=[0.0],
        )
        .segment("chirp", mode="once", phase_mode="manual")
        .tones("H")
        .use_def("tone")
        .move(df=40e6, time=4e-6, kind="linear")
        .build_resolved_ir(sample_rate_hz=625e6)
    )


def _max_line_variation(fig, ylabel_prefix: str) -> float:
    axes = [ax for ax in fig.axes if ax.get_ylabel().startswith(ylabel_prefix)]
    if not axes:
        raise AssertionError(f"Did not find axis with label prefix {ylabel_prefix!r}")
    out = 0.0
    for line in axes[0].lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            out = max(out, float(np.max(y) - np.min(y)))
    return out


def _max_abs_line_value(fig, ylabel_prefix: str) -> float:
    axes = [ax for ax in fig.axes if ax.get_ylabel().startswith(ylabel_prefix)]
    if not axes:
        raise AssertionError(f"Did not find axis with label prefix {ylabel_prefix!r}")
    out = 0.0
    for line in axes[0].lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            out = max(out, float(np.max(np.abs(y))))
    return out


def _amp_trace_variation(setup: AWGPhysicalSetupInfo) -> float:
    ir = _build_chirp_ir()

    import matplotlib

    matplotlib.use("Agg", force=True)

    fig, _axs, _slider = sequence_samples_debug(
        ir,
        physical_setup=setup,
        wait_trig_loops=1,
        include_wrap_preview=False,
        window_samples=None,
        show_slider=False,
        show_param_traces=True,
        show_spot_grid=False,
        channels=[0],
        param_channels=[0],
    )
    try:
        amp_axes = [ax for ax in fig.axes if ax.get_ylabel().startswith("CH0 amp")]
        if not amp_axes:
            raise AssertionError("Did not find CH0 amp axis in debug figure")
        amp_ax = amp_axes[0]
        variations: list[float] = []
        for line in amp_ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                variations.append(float(np.max(y) - np.min(y)))
        return max(variations) if variations else 0.0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestDebugSamples(unittest.TestCase):
    def test_param_amp_trace_reflects_channel_calibration(self) -> None:
        no_cal = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0})
        with_cal = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(
                AODSin2Calib(
                    g_poly_high_to_low=(1.0,),
                    v0_a_poly_high_to_low=(50.0, 100.0),
                    freq_min_hz=80e6,
                    freq_max_hz=120e6,
                    min_v0_sq=1e-12,
                ),
            ),
        )

        variation_no_cal = _amp_trace_variation(no_cal)
        variation_with_cal = _amp_trace_variation(with_cal)

        self.assertLess(variation_no_cal, 1e-6)
        self.assertGreater(variation_with_cal, 1.0)

    def test_amp_trace_kind_both_shows_optical_and_rf(self) -> None:
        setup = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(
                AODSin2Calib(
                    g_poly_high_to_low=(1.0,),
                    v0_a_poly_high_to_low=(50.0, 100.0),
                    freq_min_hz=80e6,
                    freq_max_hz=120e6,
                    min_v0_sq=1e-12,
                ),
            ),
        )
        ir = _build_chirp_ir()

        import matplotlib

        matplotlib.use("Agg", force=True)

        fig, _axs, _slider = sequence_samples_debug(
            ir,
            physical_setup=setup,
            wait_trig_loops=1,
            include_wrap_preview=False,
            window_samples=None,
            show_slider=False,
            show_param_traces=True,
            amp_trace_kind="both",
            show_spot_grid=False,
            channels=[0],
            param_channels=[0],
        )
        try:
            optical_var = _max_line_variation(fig, "CH0 amp opt")
            rf_var = _max_line_variation(fig, "CH0 amp rf")
            self.assertLess(optical_var, 1e-6)
            self.assertGreater(rf_var, 1.0)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_waveform_units_voltage_mv_avoids_int16_clipping(self) -> None:
        setup = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(
                AODSin2Calib(
                    g_poly_high_to_low=(1.0,),
                    v0_a_poly_high_to_low=(50.0, 100.0),
                    freq_min_hz=80e6,
                    freq_max_hz=120e6,
                    min_v0_sq=1e-12,
                ),
            ),
        )
        ir = _build_chirp_ir()

        import matplotlib

        matplotlib.use("Agg", force=True)

        fig_i16, _axs_i16, _ = sequence_samples_debug(
            ir,
            physical_setup=setup,
            wait_trig_loops=1,
            include_wrap_preview=False,
            window_samples=None,
            show_slider=False,
            show_param_traces=False,
            show_spot_grid=False,
            channels=[0],
            full_scale_mv=1.0,
            full_scale=32767,
            waveform_units="card_int16",
        )
        fig_mv, _axs_mv, _ = sequence_samples_debug(
            ir,
            physical_setup=setup,
            wait_trig_loops=1,
            include_wrap_preview=False,
            window_samples=None,
            show_slider=False,
            show_param_traces=False,
            show_spot_grid=False,
            channels=[0],
            waveform_units="voltage_mV",
        )
        try:
            peak_i16 = _max_abs_line_value(fig_i16, "CH0")
            peak_mv = _max_abs_line_value(fig_mv, "CH0")
            self.assertGreater(peak_i16, 30000.0)
            self.assertLess(peak_mv, 500.0)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig_i16)
            plt.close(fig_mv)
