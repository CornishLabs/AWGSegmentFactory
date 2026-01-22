"""
Example: sample-level debug plot with segment boundaries.

In Jupyter, you may want:
  %matplotlib widget
"""

from __future__ import annotations

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.debug import sequence_samples_debug


def _maybe_enable_matplotlib_widget_backend() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None or getattr(ip, "kernel", None) is None:
            return
        ip.run_line_magic("matplotlib", "widget")
    except Exception:
        return


def main() -> None:
    _maybe_enable_matplotlib_widget_backend()
    fs = 625e6

    ir = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define("H0", logical_channel="H", freqs=[90e6], amps=[0.3], phases="auto")
        .define("V0", logical_channel="V", freqs=[100e6], amps=[0.3], phases="auto")
        .segment("wait", mode="wait_trig")
        .tones("H")
        .use_def("H0")
        .tones("V")
        .use_def("V0")
        .hold(time=200e-6)
        .segment("chirp", mode="once")
        .tones("H")
        .move(df=+2e6, time=50e-6, idxs=[0], kind="linear")
        .segment("amp_ramp_up", mode="once")
        .tones("H")
        .ramp_amp_to(amps=0.8, time=80e-6, kind="exp", tau=20e-6)
        .segment("amp_ramp_down", mode="once")
        .tones("H")
        .ramp_amp_to(amps=0.3, time=80e-6, kind="exp", tau=20e-6)
        .segment("looped_hold", mode="loop_n", loop=5)
        .hold(time=40e-6)
        .segment("wait2", mode="wait_trig")
        .hold(time=200e-6)
        .build_resolved_ir(sample_rate_hz=fs)
    )

    fig, axs, slider = sequence_samples_debug(
        ir,
        logical_channel_to_hardware_channel={"H": 0, "V": 1},
        wait_trig_loops=3,
        include_wrap_preview=True,
        window_samples=None,  # show the whole unrolled sequence; zoom in with the GUI
        show_slider=False,
        show_markers=False,
        title="AWG samples",
    )

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
