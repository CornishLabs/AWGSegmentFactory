from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .timeline import ResolvedTimeline

@dataclass(frozen=True)
class LinearFreqToPos:
    f0_hz: float = 0.0
    slope_hz_per_unit: float = 1e6  # 1 unit per MHz by default

    def __call__(self, f_hz: np.ndarray) -> np.ndarray:
        return (f_hz - self.f0_hz) / self.slope_hz_per_unit

def default_size_fn(aH: np.ndarray, aV: np.ndarray) -> np.ndarray:
    # outer product -> flattened NxM sizes
    A = np.sqrt(np.clip(aH, 0, None)[:, None] * np.clip(aV, 0, None)[None, :])
    return 40.0 * A.reshape(-1)

def interactive_grid_debug(
    tl: ResolvedTimeline,
    *,
    plane_h: str = "H",
    plane_v: str = "V",
    fx: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    size_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_size_fn,
    fps: float = 200.0,
    title: str = "Crossed-AOD Grid Debug",
    annotate: bool = True,
):
    """
    Jupyter interactive plot. Shows NxM traps as Cartesian product of H and V tones.
    Requires ipywidgets in your notebook environment.
    """
    if fx is None:
        fx = LinearFreqToPos(slope_hz_per_unit=1e6)  # MHz
    if fy is None:
        fy = LinearFreqToPos(slope_hz_per_unit=1e6)  # MHz

    times = tl.sample_times(fps=fps)

    import ipywidgets as widgets
    from IPython.display import display

    fig, ax = plt.subplots()

    out = widgets.Output()

    def render_frame(i: int):
        t = float(times[i])
        sH = tl.state_at(plane_h, t)
        sV = tl.state_at(plane_v, t)

        xH = fx(sH.freqs_hz)  # (N,)
        yV = fy(sV.freqs_hz)  # (M,)

        # Cartesian product points (N*M)
        X = np.repeat(xH, len(yV))
        Y = np.tile(yV, len(xH))
        S = size_fn(sH.amps, sV.amps)

        ax.clear()
        ax.set_title(f"{title} | t={t*1e3:.3f} ms")
        ax.set_xlabel(f"{plane_h} position (arb)")
        ax.set_ylabel(f"{plane_v} position (arb)")
        ax.grid(True)
        ax.scatter(X, Y, s=S, marker='o')

        # microns
        X_LIM = (-60, 60)
        Y_LIM = (-60, 60)

        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.set_aspect("equal", adjustable="box")  # keeps geometry sensible

        if annotate:
            # label as (i,j)
            k = 0
            for ii in range(len(xH)):
                for jj in range(len(yV)):
                    ax.text(X[k], Y[k], f"{ii},{jj}", fontsize=8)
                    k += 1

        fig.canvas.draw_idle()

    play = widgets.Play(
        interval=int(1000 / fps),
        value=0,
        min=0,
        max=len(times) - 1,
        step=1,
        description="Play",
    )
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(times) - 1,
        step=1,
        description="frame",
        continuous_update=True,
        readout=False,
        layout=widgets.Layout(width="70%"),
    )
    widgets.jslink((play, "value"), (slider, "value"))

    def on_change(change):
        if change["name"] == "value":
            with out:
                render_frame(change["new"])

    slider.observe(on_change)
    display(widgets.HBox([play, slider]))
    display(out)

    with out:
        render_frame(0)



    return fig, ax
