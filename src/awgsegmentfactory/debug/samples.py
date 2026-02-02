"""Sample-level debug/inspection helpers.

These utilities unroll a compiled Spectrum-like sequence program into a contiguous
sample buffer and render it with segment/repeat boundary markers for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..resolved_ir import ResolvedIR
from ..synth_samples import CompiledSequenceProgram, compile_sequence_program
from ..synth_samples import (
    _interp_logical_channel_part as _interp_logical_channel_part,
)
from ..quantize import QuantizedIR, quantize_resolved_ir
from ..types import ChannelMap


@dataclass(frozen=True)
class SegmentInstance:
    """One occurrence of a segment during debug-unrolled playback (with boundaries)."""

    start_sample: int
    stop_sample: int
    step_index: int
    segment_index: int
    segment_name: str
    rep_index: int
    reps_total: int
    is_wrap_preview: bool = False


def _iter_instances_for_debug(
    compiled: CompiledSequenceProgram,
    *,
    wait_trig_loops: int,
    include_wrap_preview: bool,
    max_step_transitions: Optional[int],
) -> list[tuple[int, int, int, int, bool]]:
    """
    Returns a list of (step_index, segment_index, rep_index, reps_total, is_wrap_preview)
    in the order they'd be replayed.
    """
    if wait_trig_loops <= 0:
        raise ValueError("wait_trig_loops must be > 0")

    steps = {s.step_index: s for s in compiled.steps}
    if not steps:
        raise ValueError("No steps in compiled program")
    if 0 not in steps:
        raise ValueError("Expected step_index 0 to exist (entry step)")

    max_transitions = (
        max_step_transitions if max_step_transitions is not None else len(steps) + 1
    )
    if max_transitions <= 0:
        raise ValueError("max_step_transitions must be > 0")

    order: list[tuple[int, int, int, int, bool]] = []
    entry = 0
    step_idx = entry
    for _ in range(max_transitions):
        step = steps[step_idx]
        reps_total = wait_trig_loops if step.on_trig else int(step.loops)
        if reps_total <= 0:
            raise ValueError(f"Invalid loops={reps_total} for step {step.step_index}")
        for rep_index in range(reps_total):
            order.append(
                (step.step_index, step.segment_index, rep_index, reps_total, False)
            )
        step_idx = int(step.next_step)
        if step_idx == entry:
            break
    else:  # pragma: no cover
        raise RuntimeError(
            f"Sequence did not return to entry step {entry} within max_step_transitions={max_transitions}"
        )

    if include_wrap_preview:
        step0 = steps[entry]
        order.append((step0.step_index, step0.segment_index, 0, 1, True))

    return order


def unroll_compiled_sequence_for_debug(
    compiled: CompiledSequenceProgram,
    *,
    wait_trig_loops: int = 3,
    include_wrap_preview: bool = True,
    max_step_transitions: Optional[int] = None,
) -> tuple[np.ndarray, Tuple[SegmentInstance, ...]]:
    """
    Unroll a compiled sequence into a contiguous sample array for debugging.

    Notes:
    - `wait_trig` steps are unrolled as `wait_trig_loops` repeats (as if a trigger
      happened right after).
    - `loop_n` steps use their programmed `loops`.
    - If `include_wrap_preview=True`, appends the entry step once so the final
      wrap-around boundary is visible.
    """
    n_channels = max(compiled.logical_channel_to_hardware_channel.values()) + 1
    instances_spec = _iter_instances_for_debug(
        compiled,
        wait_trig_loops=wait_trig_loops,
        include_wrap_preview=include_wrap_preview,
        max_step_transitions=max_step_transitions,
    )

    seg_lens = [int(s.n_samples) for s in compiled.segments]
    total = sum(
        seg_lens[seg_idx] for _step, seg_idx, _rep, _rep_total, _wrap in instances_spec
    )
    out = np.zeros((n_channels, total), dtype=np.int16)

    instances: list[SegmentInstance] = []
    cursor = 0
    for step_idx, seg_idx, rep_index, reps_total, is_wrap_preview in instances_spec:
        seg = compiled.segments[seg_idx]
        n = int(seg.n_samples)
        out[:, cursor : cursor + n] = seg.data_i16
        instances.append(
            SegmentInstance(
                start_sample=cursor,
                stop_sample=cursor + n,
                step_index=step_idx,
                segment_index=seg_idx,
                segment_name=str(seg.name),
                rep_index=rep_index,
                reps_total=reps_total,
                is_wrap_preview=is_wrap_preview,
            )
        )
        cursor += n

    return out, tuple(instances)


def sequence_samples_debug(
    program: ResolvedIR | QuantizedIR | CompiledSequenceProgram,
    *,
    logical_channel_to_hardware_channel: Optional[ChannelMap] = None,
    wait_trig_loops: int = 3,
    include_wrap_preview: bool = True,
    gain: float = 1.0,
    clip: float = 0.9,
    full_scale: int = 32767,
    channels: Optional[Sequence[int]] = None,
    window_samples: Optional[int] = None,
    show_slider: Optional[bool] = None,
    show_markers: bool = True,
    show_param_traces: bool = True,
    param_channels: Optional[Sequence[int]] = None,
    freq_unit: str = "MHz",
    max_points_wave: int = 20_000,
    max_points_param: int = 10_000,
    title: str = "Sequence Samples Debug",
):
    """
    Interactive sample-level debug view with segment boundaries.

    - Plots raw int16 samples per enabled channel (with dynamic downsampling for speed).
    - Draws vertical boundary markers at every segment repetition (including loops):
      solid = transition to a different segment, dashed = repeat of the same segment.
    - Optional per-sample frequency/amplitude traces (if `show_param_traces=True`) are
      rendered on the same unrolled x-axis (including `wait_trig_loops` repeats).
    - Optional boundary slider (index into the boundary list) that re-centers the view.
    - Set `window_samples=None` to plot the entire unrolled sequence at once (no slider).

    Tip (Jupyter): `%matplotlib widget` for a nicer zoom/pan UI.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import Slider
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "`sequence_samples_debug` requires matplotlib. Install the `dev` dependency group."
        ) from exc

    compiled: CompiledSequenceProgram
    q_ir: Optional[ResolvedIR] = None
    if isinstance(program, CompiledSequenceProgram):
        compiled = program
    elif isinstance(program, QuantizedIR):
        q_ir = program.resolved_ir
        compiled = compile_sequence_program(
            program, gain=gain, clip=clip, full_scale=full_scale
        )
    else:
        if logical_channel_to_hardware_channel is None:
            raise ValueError(
                "logical_channel_to_hardware_channel is required when passing a ResolvedIR"
            )
        quantized = quantize_resolved_ir(
            program,
            logical_channel_to_hardware_channel=logical_channel_to_hardware_channel,
        )
        q_ir = quantized.resolved_ir
        compiled = compile_sequence_program(
            quantized,
            gain=gain,
            clip=clip,
            full_scale=full_scale,
        )
    if show_param_traces and q_ir is None:
        raise ValueError(
            "show_param_traces=True requires passing a ResolvedIR/QuantizedIR (not a CompiledSequenceProgram)"
        )

    samples, instances = unroll_compiled_sequence_for_debug(
        compiled,
        wait_trig_loops=wait_trig_loops,
        include_wrap_preview=include_wrap_preview,
    )
    total = int(samples.shape[1])
    full_view = window_samples is None or window_samples >= total
    if not full_view and window_samples <= 0:  # type: ignore[operator]
        raise ValueError("window_samples must be > 0 when provided")
    if show_slider is None:
        show_slider = not full_view
    if full_view:
        show_slider = False

    if max_points_wave <= 2:
        raise ValueError("max_points_wave must be > 2")
    if max_points_param <= 2:
        raise ValueError("max_points_param must be > 2")

    if channels is None:
        channels = sorted(set(compiled.logical_channel_to_hardware_channel.values()))
    channels = [int(c) for c in channels]
    if not channels:
        raise ValueError("channels must be non-empty")
    if min(channels) < 0 or max(channels) >= samples.shape[0]:
        raise ValueError("channels out of range for compiled data")

    # Invert logical_channel->hardware_channel mapping for labels (best-effort).
    hw_ch_to_logical_channels: Dict[int, List[str]] = {}
    for logical_channel, hw_ch in compiled.logical_channel_to_hardware_channel.items():
        hw_ch_to_logical_channels.setdefault(int(hw_ch), []).append(str(logical_channel))

    if param_channels is None:
        param_channels = list(channels)
    param_channels = [int(c) for c in param_channels]
    if show_param_traces:
        param_channels = param_channels[:2]

    n_wave = len(channels)
    n_param_channels = 0
    if show_param_traces:
        n_param_channels = min(2, len(param_channels))
    n_param_axes = 2 * n_param_channels

    # Vertical stack layout: frequency/amp traces (optional) above waveform channels,
    # all sharing the same x-axis.
    fig = plt.figure(figsize=(14, 2.4 * max(1, n_param_channels) + 2.6 * n_wave))
    gs = GridSpec(
        nrows=n_param_axes + n_wave,
        ncols=1,
        figure=fig,
        height_ratios=[1] * n_param_axes + [2] * n_wave,
    )

    ax_master = None
    ax_param: list = []
    param_axes: list[tuple[int, object, object]] = []
    row = 0
    if show_param_traces and n_param_channels > 0:
        for ch in param_channels[:n_param_channels]:
            if ax_master is None:
                ax_f = fig.add_subplot(gs[row, 0])
                ax_master = ax_f
            else:
                ax_f = fig.add_subplot(gs[row, 0], sharex=ax_master)
            row += 1
            ax_a = fig.add_subplot(gs[row, 0], sharex=ax_master)
            row += 1
            ax_param.extend([ax_f, ax_a])
            param_axes.append((int(ch), ax_f, ax_a))

    axs = []
    for i in range(n_wave):
        r = row + i
        if ax_master is None:
            ax = fig.add_subplot(gs[r, 0])
            ax_master = ax
        else:
            ax = fig.add_subplot(gs[r, 0], sharex=ax_master)
        axs.append(ax)

    # With shared x-axis, only show x tick labels on the bottom-most axis.
    for ax in ax_param + axs[:-1]:
        ax.tick_params(labelbottom=False)

    boundary_positions = [inst.start_sample for inst in instances]
    slider = None
    if show_slider:
        ax_slider = fig.add_axes((0.15, 0.02, 0.7, 0.03))
        slider = Slider(
            ax_slider,
            "boundary",
            valmin=0,
            valmax=max(0, len(boundary_positions) - 1),
            valinit=0,
            valstep=1,
        )

    # Plot objects (windowed view).
    wave_lines = []
    selected_lines = []

    # Parameter line objects (tone lines). Created lazily on first render.
    param_lines_freq: Dict[int, list] = {}
    param_lines_amp: Dict[int, list] = {}
    segment_param_cache: Dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    max_tones_cache: Dict[str, int] = {}

    def _window_for_boundary(sample_pos: int) -> tuple[int, int]:
        """Compute the [x0, x1) window centered on a boundary sample index."""
        if window_samples is None:
            return 0, total
        half = window_samples // 2
        x0 = max(0, int(sample_pos) - half)
        x1 = min(total, int(sample_pos) + half)
        if x1 - x0 < window_samples and total >= window_samples:
            # Prefer fixed-width window when possible.
            if x0 == 0:
                x1 = window_samples
            elif x1 == total:
                x0 = total - window_samples
        return x0, x1

    for ax, ch in zip(axs, channels):
        marker = "." if show_markers else ""
        markersize = 2 if show_markers else 0
        (line,) = ax.plot(
            [], [], linestyle="-", linewidth=0.8, marker=marker, markersize=markersize
        )
        wave_lines.append(line)

        logical_channels = ",".join(sorted(hw_ch_to_logical_channels.get(ch, [])))
        label = f"CH{ch}" + (f" ({logical_channels})" if logical_channels else "")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

        if not full_view:
            sel = ax.axvline(
                boundary_positions[0] if boundary_positions else 0,
                color="C3",
                linewidth=1.5,
                alpha=0.8,
            )
            selected_lines.append(sel)

    axs[-1].set_xlabel("Sample index (per-sample)")
    y_max = float(compiled.clip) * float(compiled.full_scale)
    if y_max > 0:
        for ax in axs:
            ax.set_ylim(-1.05 * y_max, 1.05 * y_max)

    # One selected-boundary line per param axis too (windowed/slider mode only).
    if show_param_traces and not full_view:
        for ax in ax_param:
            sel = ax.axvline(
                boundary_positions[0] if boundary_positions else 0,
                color="C3",
                linewidth=1.5,
                alpha=0.8,
            )
            selected_lines.append(sel)

    if show_param_traces:
        # Configure param axes titles/labels.
        if freq_unit not in ("Hz", "kHz", "MHz", "GHz"):
            raise ValueError("freq_unit must be one of: 'Hz', 'kHz', 'MHz', 'GHz'")
        f_scale = {"Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9}[freq_unit]

        def _param_logical_channel_for_hardware_channel(ch: int) -> Optional[str]:
            logical_channels = sorted(hw_ch_to_logical_channels.get(ch, []))
            return logical_channels[0] if logical_channels else None

        def _max_tones_for_logical_channel(logical_channel: str) -> int:
            if logical_channel in max_tones_cache:
                return max_tones_cache[logical_channel]
            if q_ir is None:
                return 0
            m = 0
            for seg in q_ir.segments:
                for part in seg.parts:
                    pp = part.logical_channels[logical_channel]
                    m = max(m, int(pp.start.freqs_hz.shape[0]))
            max_tones_cache[logical_channel] = m
            return m

        def _segment_params(
            seg_index: int, logical_channel: str
        ) -> tuple[np.ndarray, np.ndarray]:
            key = (int(seg_index), logical_channel)
            if key in segment_param_cache:
                return segment_param_cache[key]
            if q_ir is None:
                raise RuntimeError("q_ir missing")
            seg = q_ir.segments[int(seg_index)]
            n = int(seg.n_samples)
            max_tones = _max_tones_for_logical_channel(logical_channel)
            freqs = np.full((n, max_tones), np.nan, dtype=np.float32)
            amps = np.full((n, max_tones), np.nan, dtype=np.float32)
            cursor = 0
            for part in seg.parts:
                pp = part.logical_channels[logical_channel]
                if part.n_samples <= 0:
                    continue
                f_part, a_part = _interp_logical_channel_part(
                    pp, n_samples=part.n_samples, sample_rate_hz=q_ir.sample_rate_hz
                )
                f_part = f_part.astype(np.float32, copy=False)
                a_part = a_part.astype(np.float32, copy=False)
                nt = int(f_part.shape[1])
                freqs[cursor : cursor + part.n_samples, :nt] = f_part
                amps[cursor : cursor + part.n_samples, :nt] = a_part
                cursor += part.n_samples
            segment_param_cache[key] = (freqs, amps)
            return freqs, amps

        for idx, (ch, ax_f, ax_a) in enumerate(param_axes):
            ax_f.set_ylabel(f"CH{ch} freq ({freq_unit})")
            ax_a.set_ylabel(f"CH{ch} amp")
            ax_f.grid(True, alpha=0.25)
            ax_a.grid(True, alpha=0.25)
            if idx == 0:
                ax_f.set_title("Frequency vs time")
                ax_a.set_title("Amplitude vs time")

    def _boundary_label(i: int) -> str:
        """Return a human-friendly label describing boundary `i` for the figure title."""
        if not instances:
            return title
        cur = instances[i]
        if i == 0:
            return f"{title} | start: {cur.segment_name}"
        prev = instances[i - 1]
        if cur.is_wrap_preview:
            return f"{title} | wrap: {prev.segment_name} -> {cur.segment_name}"
        if prev.segment_index == cur.segment_index:
            return f"{title} | loop: {cur.segment_name} ({cur.rep_index + 1}/{cur.reps_total})"
        return f"{title} | seg: {prev.segment_name} -> {cur.segment_name}"

    # Fast boundary rendering: one LineCollection per axis for solid + dashed.
    boundary_solid: list[float] = []
    boundary_repeat: list[float] = []
    for j, inst in enumerate(instances):
        pos = float(inst.start_sample)
        if (
            j > 0
            and instances[j - 1].segment_index == inst.segment_index
            and not inst.is_wrap_preview
        ):
            boundary_repeat.append(pos)
        else:
            boundary_solid.append(pos)

    def _vline_segments(xs: Sequence[float]) -> np.ndarray:
        """Build axis-coordinate segments for a `LineCollection` vertical line overlay."""
        if not xs:
            return np.zeros((0, 2, 2), dtype=float)
        x = np.asarray(xs, dtype=float)
        segs = np.zeros((x.size, 2, 2), dtype=float)
        segs[:, 0, 0] = x
        segs[:, 1, 0] = x
        segs[:, 0, 1] = 0.0
        segs[:, 1, 1] = 1.0
        return segs

    for ax in ax_param + axs:
        if boundary_solid:
            lc_solid = LineCollection(
                _vline_segments(boundary_solid),
                colors="k",
                linewidths=1.2,
                linestyles="-",
                alpha=0.55,
                transform=ax.get_xaxis_transform(),
            )
            ax.add_collection(lc_solid)
        if boundary_repeat:
            lc_rep = LineCollection(
                _vline_segments(boundary_repeat),
                colors="k",
                linewidths=1.0,
                linestyles="--",
                alpha=0.35,
                transform=ax.get_xaxis_transform(),
            )
            ax.add_collection(lc_rep)

    # Precompute arrays for fast mapping from global sample index -> instance.
    inst_starts = np.array([inst.start_sample for inst in instances], dtype=int)
    inst_stops = np.array([inst.stop_sample for inst in instances], dtype=int)

    def _downsample_step(n: int, *, max_points: int) -> int:
        """Choose a stride so plotting uses at most `max_points` points."""
        if n <= max_points:
            return 1
        return int(np.ceil(n / max_points))

    def _wave_view(ch: int, *, x0w: int, x1w: int) -> tuple[np.ndarray, np.ndarray]:
        """Return downsampled (x, y) waveform data for channel `ch` over [x0w, x1w)."""
        n = max(0, int(x1w) - int(x0w))
        if n <= 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.int16)
        step = _downsample_step(n, max_points=max_points_wave)
        x = np.arange(x0w, x1w, step, dtype=int)
        y = samples[ch, x0w:x1w:step]
        return x, y

    def _params_at(logical_channel: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (freqs, amps) at global sample indices `x` for one logical channel."""
        if q_ir is None:
            raise RuntimeError("q_ir missing")
        if x.size == 0:
            max_tones = _max_tones_for_logical_channel(logical_channel)
            z = np.full((0, max_tones), np.nan, dtype=np.float32)
            return z, z

        inst_idx = np.searchsorted(inst_stops, x, side="right")
        inst_idx = np.clip(inst_idx, 0, len(instances) - 1)
        local = x - inst_starts[inst_idx]

        max_tones = _max_tones_for_logical_channel(logical_channel)
        freqs = np.full((x.size, max_tones), np.nan, dtype=np.float32)
        amps = np.full((x.size, max_tones), np.nan, dtype=np.float32)

        for idx in np.unique(inst_idx):
            mask = inst_idx == idx
            seg_index = int(instances[int(idx)].segment_index)
            seg_freqs, seg_amps = _segment_params(seg_index, logical_channel)
            offs = local[mask]
            freqs[mask, :] = seg_freqs[offs, :]
            amps[mask, :] = seg_amps[offs, :]
        return freqs, amps

    def _set_view(i: int) -> None:
        """Set x-limits and selection markers to the boundary at index `i` (slider mode)."""
        i = int(i)
        i = max(0, min(i, len(boundary_positions) - 1))
        pos = boundary_positions[i]
        x0w, x1w = _window_for_boundary(pos)
        for ax in ax_param + axs:
            ax.set_xlim(x0w, x1w - 1 if x1w > x0w else x1w)

        # selected_lines contains first the wave axes, then param axes (if enabled).
        for sel in selected_lines:
            sel.set_xdata([pos, pos])

        fig.suptitle(_boundary_label(i))
        fig.canvas.draw_idle()

    updating = False

    def _update_from_xlim(_ax=None) -> None:
        """Update plotted data after pan/zoom (dynamic downsampling for performance)."""
        nonlocal updating
        if updating:
            return
        updating = True
        try:
            xlim = axs[-1].get_xlim()
            x0w = max(0, int(np.floor(xlim[0])))
            x1w = min(total, int(np.ceil(xlim[1])))
            if x1w <= x0w:
                return

            for line, ch in zip(wave_lines, channels):
                x, y = _wave_view(int(ch), x0w=x0w, x1w=x1w)
                line.set_data(x, y)

            if show_param_traces:
                n = x1w - x0w
                step = _downsample_step(n, max_points=max_points_param)
                x = np.arange(x0w, x1w, step, dtype=int)
                x_float = x.astype(float, copy=False)
                for ch, ax_f, ax_a in param_axes:
                    logical_channel = _param_logical_channel_for_hardware_channel(ch)
                    if logical_channel is None:
                        continue
                    freqs_w, amps_w = _params_at(logical_channel, x)
                    freqs_plot = freqs_w * f_scale

                    lf = param_lines_freq.setdefault(ch, [])
                    la = param_lines_amp.setdefault(ch, [])
                    nt = int(freqs_plot.shape[1])
                    while len(lf) < nt:
                        (ln,) = ax_f.plot([], [], linewidth=1.0, alpha=0.9)
                        lf.append(ln)
                    while len(la) < nt:
                        (ln,) = ax_a.plot([], [], linewidth=1.0, alpha=0.9)
                        la.append(ln)

                    for k in range(nt):
                        lf[k].set_data(x_float, freqs_plot[:, k])
                        la[k].set_data(x_float, amps_w[:, k])
                        lf[k].set_visible(np.any(np.isfinite(freqs_plot[:, k])))
                        la[k].set_visible(np.any(np.isfinite(amps_w[:, k])))
                    for k in range(nt, len(lf)):
                        lf[k].set_visible(False)
                    for k in range(nt, len(la)):
                        la[k].set_visible(False)

                    finite = freqs_plot[np.isfinite(freqs_plot)]
                    if finite.size:
                        lo = float(np.min(finite))
                        hi = float(np.max(finite))
                        if lo == hi:
                            m = 1.0 if lo == 0.0 else abs(lo) * 0.01
                        else:
                            m = 0.05 * (hi - lo)
                        ax_f.set_ylim(lo - m, hi + m)

                    finite_a = amps_w[np.isfinite(amps_w)]
                    if finite_a.size:
                        lo = float(np.min(finite_a))
                        hi = float(np.max(finite_a))
                        m = 0.05 * max(abs(lo), abs(hi))
                        if m == 0.0:
                            m = 0.05
                        ax_a.set_ylim(lo - m, hi + m)
        finally:
            updating = False

    if full_view:
        fig.suptitle(title)
        for ax in ax_param + axs:
            ax.set_xlim(0, total - 1 if total > 0 else 0)
        for ax in ax_param + axs:
            ax.callbacks.connect("xlim_changed", _update_from_xlim)
        _update_from_xlim()
        fig.subplots_adjust(bottom=0.06, top=0.93, hspace=0.35, wspace=0.25)
        return fig, axs, None

    fig.suptitle(_boundary_label(0))
    for ax in ax_param + axs:
        ax.callbacks.connect("xlim_changed", _update_from_xlim)
    _set_view(0)

    if slider is not None:

        def _on_slider_change(val):
            _set_view(int(val))

        slider.on_changed(_on_slider_change)

        # Keep widgets alive even if the caller ignores the return value.
        fig._awgsegmentfactory_widgets = getattr(
            fig, "_awgsegmentfactory_widgets", []
        ) + [slider]  # type: ignore[attr-defined]

    fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.35, wspace=0.25)
    return fig, axs, slider
