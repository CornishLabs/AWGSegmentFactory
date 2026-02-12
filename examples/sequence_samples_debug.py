"""
Example: sample-level debug plot with segment boundaries.

In Jupyter, you may want:
  %matplotlib widget
"""

from __future__ import annotations

from awgsegmentfactory import AWGPhysicalSetupInfo, AWGProgramBuilder, ResolvedIR
from awgsegmentfactory.debug import LinearFreqToPos, sequence_samples_debug


from awgsegmentfactory.presets import recreate_mol_exp


def _maybe_enable_matplotlib_widget_backend() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None or getattr(ip, "kernel", None) is None:
            return
        ip.run_line_magic("matplotlib", "widget")
    except Exception:
        return

def _build_demo_program(
    *, sample_rate_hz: float
) -> ResolvedIR:
    b = recreate_mol_exp()

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)

def main() -> None:
    _maybe_enable_matplotlib_widget_backend()
    fs = 625e6

    ir = _build_demo_program(sample_rate_hz=fs)
    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})

    # Optional: map frequency to "position units" for a 2D spot-grid view.
    fx = LinearFreqToPos(f0_hz=100e6, slope_hz_per_unit=250e3)  # e.g. µm if 250 kHz/µm
    fy = LinearFreqToPos(f0_hz=100e6, slope_hz_per_unit=-250e3)

    fig, axs, slider = sequence_samples_debug(
        ir,
        physical_setup=physical_setup,
        wait_trig_loops=3,
        include_wrap_preview=True,
        window_samples=None,  # show the whole unrolled sequence; zoom in with the GUI
        show_slider=False,
        show_markers=False,
        show_spot_grid=True,
        spot_grid_fx=fx,
        spot_grid_fy=fy,
        spot_grid_annotate=False,
        title="AWG samples",
    )

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
