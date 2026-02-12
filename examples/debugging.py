# %%
from __future__ import annotations

import numpy as np

from awgsegmentfactory import AWGProgramBuilder

from awgsegmentfactory.presets import recreate_mol_exp

# In a notebook, enable the ipympl widget backend for interactive plots.
# When run as plain Python, this is a no-op (so the file stays runnable/testable).
def _maybe_enable_matplotlib_widget_backend() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None or getattr(ip, "kernel", None) is None:
            return
        ip.run_line_magic("matplotlib", "widget")
    except Exception:
        # If ipympl isn't installed (or we're not in a notebook), just proceed with
        # the default backend so this file remains runnable as a plain script.
        return


_maybe_enable_matplotlib_widget_backend()
# %%
fs = 625e6
prog = recreate_mol_exp().build_timeline(sample_rate_hz=fs)

# %%
try:
    from IPython import get_ipython  # type: ignore

    ip = get_ipython()
    _IN_JUPYTER = ip is not None and getattr(ip, "kernel", None) is not None
except Exception:
    _IN_JUPYTER = False

if _IN_JUPYTER:
    from awgsegmentfactory.debug import interactive_grid_debug, LinearFreqToPos

    # Optional: map frequency to "position units" for nicer axes.
    fx = LinearFreqToPos(f0_hz=100e6, slope_hz_per_unit=250e3)  # e.g. µm if 250 kHz/µm
    fy = LinearFreqToPos(f0_hz=100e6, slope_hz_per_unit=-250e3)

    fig, ax = interactive_grid_debug(
        prog,
        logical_channel_h="H",
        logical_channel_v="V",
        fx=fx,
        fy=fy,
        fps=30*1e4,
        annotate=False,  # turn on if you want (i,j) labels
    )
else:
    print(
        "Not running in a Jupyter kernel; skipping interactive plot (interactive_grid_debug)."
    )
# %%
