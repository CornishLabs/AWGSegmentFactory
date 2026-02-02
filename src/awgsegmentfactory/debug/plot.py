"""High-level plotting helpers for inspecting `ResolvedTimeline` programs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import bisect


import numpy as np

from ..resolved_timeline import ResolvedTimeline


@dataclass(frozen=True)
class LinearFreqToPos:
    """Simple frequency→position mapping used by debug plots (linear scale/offset)."""

    f0_hz: float = 0.0
    slope_hz_per_unit: float = 1e6  # 1 unit per MHz by default

    def __call__(self, f_hz: np.ndarray) -> np.ndarray:
        """Convert frequencies in Hz into position units."""
        return (f_hz - self.f0_hz) / self.slope_hz_per_unit


def norm_total_amp_fn(aH: np.ndarray, aV: np.ndarray) -> np.ndarray:
    A = ((np.clip(aH, 0, None)[:, None] * np.clip(aV, 0, None)[None, :]))**(1/4)
    return A.reshape(-1)

def default_size_fn(aH: np.ndarray, aV: np.ndarray) -> np.ndarray:
    """Default marker size function for NxM grids (geometric mean of H/V amplitudes)."""
    # outer product -> flattened NxM sizes
    A = np.sqrt(norm_total_amp_fn(aH, aV))
    return 60.0 * A


def interactive_grid_debug(
    tl: ResolvedTimeline,
    *,
    logical_channel_h: str = "H",
    logical_channel_v: str = "V",
    fx: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    size_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_size_fn,
    fps: float = 200.0,
    title: str = "Crossed-AOD Grid Debug",
    annotate: bool = True,
    show_segment_in_title: bool = True,
):
    """
    Jupyter interactive plot. Shows NxM traps as Cartesian product of H and V tones.
    Requires ipywidgets in your notebook environment.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "`interactive_grid_debug` requires matplotlib (and ipywidgets). Install the `dev` dependency group."
        ) from exc

    if fx is None:
        fx = LinearFreqToPos(slope_hz_per_unit=1e6)  # MHz
    if fy is None:
        fy = LinearFreqToPos(slope_hz_per_unit=1e6)  # MHz

    # Allow passing ResolvedIR directly for convenience.
    if hasattr(tl, "to_timeline"):
        tl = tl.to_timeline()  # type: ignore[assignment]

    times = tl.sample_times(fps=fps)

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "`interactive_grid_debug` requires ipywidgets. Install the `dev` dependency group."
        ) from exc

    fig, ax = plt.subplots()

    out = widgets.Output()

    def _axis_limits_for_logical_channel(
        logical_channel: str, f_to_pos: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[float, float]:
        """Compute padded axis limits from all span endpoints for a logical channel."""
        spans = tl.spans_by_logical_channel.get(logical_channel, [])
        if not spans:
            return (-1.0, 1.0)
        freqs: list[np.ndarray] = []
        for sp in spans:
            freqs.append(sp.start.freqs_hz)
            freqs.append(sp.end.freqs_hz)
        f = (
            np.concatenate([x for x in freqs if x.size], axis=0)
            if any(x.size for x in freqs)
            else np.zeros((0,))
        )
        if f.size == 0:
            return (-1.0, 1.0)
        x = f_to_pos(f)
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmin == xmax:
            m = 1.0 if xmin == 0.0 else abs(xmin) * 0.05
        else:
            m = 0.05 * (xmax - xmin)
        return (xmin - m, xmax + m)

    xlim = _axis_limits_for_logical_channel(logical_channel_h, fx)
    ylim = _axis_limits_for_logical_channel(logical_channel_v, fy)

    start_times = [t0 for (t0, _name) in tl.segment_starts]
    start_names = [_name for (_t0, _name) in tl.segment_starts]

    def _segment_name_at(t: float) -> str:
        """Return the segment name active at time `t` (based on `segment_starts`)."""
        if not start_times:
            return ""
        i = bisect.bisect_right(start_times, t) - 1
        if i < 0:
            return start_names[0]
        return start_names[min(i, len(start_names) - 1)]

    def render_frame(i: int):
        """Render frame `i` (a time index) into the matplotlib axes."""
        t = float(times[i])
        sH = tl.state_at(logical_channel_h, t)
        sV = tl.state_at(logical_channel_v, t)

        xH = fx(sH.freqs_hz)  # (N,)
        yV = fy(sV.freqs_hz)  # (M,)

        # Cartesian product points (N*M)
        X = np.repeat(xH, len(yV))
        Y = np.tile(yV, len(xH))
        S = size_fn(sH.amps, sV.amps)
        Alpha = norm_total_amp_fn(sH.amps, sV.amps)

        ax.clear()
        if show_segment_in_title:
            seg_name = _segment_name_at(t)
            ax.set_title(f"{title} | {seg_name} | t={t * 1e3:.3f} ms")
        else:
            ax.set_title(f"{title} | t={t * 1e3:.3f} ms")
        ax.set_xlabel(f"{logical_channel_h} position (arb)")
        ax.set_ylabel(f"{logical_channel_v} position (arb)")
        ax.grid(True)
        
        c='blue'
        rgba = mcolors.to_rgba_array(c)          # shape (N, 4) if c is N colors, or (1, 4) if single
        rgba = np.broadcast_to(rgba, (len(Alpha), 4)).copy()
        rgba[:, 3] = Alpha        

        ax.scatter(X, Y, s=S, marker="o", linewidth=0, edgecolor='none', color=rgba)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
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
        """ipywidgets observer: re-render when the slider value changes."""
        if change["name"] == "value":
            with out:
                render_frame(change["new"])

    slider.observe(on_change)
    display(widgets.HBox([play, slider]))
    display(out)

    with out:
        render_frame(0)

    return fig, ax
