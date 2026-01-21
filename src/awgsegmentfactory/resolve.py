from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings

from .ir import (
    ProgramSpec, SegmentSpec, SegmentMode,
    HoldOp, UseDefOp, MoveOp, RampAmpToOp, RemapFromDefOp,
)
from .timeline import PlaneState, Span, ResolvedTimeline

def _empty_state() -> PlaneState:
    return PlaneState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )

def _round_to_samples(sample_rate_hz: float, time_s: float) -> float:
    if time_s <= 0:
        return 0.0
    dt = 1.0 / sample_rate_hz
    n = int(np.ceil(time_s / dt))
    return n * dt

def _snap_freqs_to_wrap(freqs_hz: np.ndarray, seg_len_s: float) -> np.ndarray:
    # wrap-continuous if f * seg_len_s is an integer number of cycles
    if seg_len_s <= 0:
        return freqs_hz
    k = np.round(freqs_hz * seg_len_s)
    return k / seg_len_s

def _select_idxs(n: int, idxs: Optional[Tuple[int, ...]]) -> np.ndarray:
    if idxs is None:
        return np.arange(n, dtype=int)
    idx = np.array(list(idxs), dtype=int)
    if np.any(idx < 0) or np.any(idx >= n):
        raise IndexError(f"idxs out of range for n={n}: {idxs}")
    return idx

def resolve_program(spec: ProgramSpec) -> ResolvedTimeline:
    fs = spec.sample_rate_hz

    # current per-plane state
    cur: Dict[str, PlaneState] = {p: _empty_state() for p in spec.planes}
    spans: Dict[str, List[Span]] = {p: [] for p in spec.planes}

    t = 0.0
    segment_starts: List[tuple[float, str]] = []

    for seg in spec.segments:
        segment_starts.append((t, seg.name))
        seg_time_accum = 0.0

        for op in seg.ops:
            if isinstance(op, UseDefOp):
                d = spec.definitions[op.def_name]
                if d.plane != op.plane:
                    raise ValueError(f"Definition {d.name} is for plane {d.plane}, not {op.plane}")

                cur[op.plane] = PlaneState(
                    freqs_hz=np.array(d.freqs_hz, dtype=float),
                    amps=np.array(d.amps, dtype=float),
                    phases_rad=np.array(d.phases_rad, dtype=float),
                )
                continue

            if isinstance(op, RemapFromDefOp):
                d = spec.definitions[op.target_def]
                if d.plane != op.plane:
                    raise ValueError(f"Definition {d.name} is for plane {d.plane}, not {op.plane}")

                start = cur[op.plane]
                # build target arrays at dst indices
                tf = np.array(d.freqs_hz, dtype=float)[list(op.dst)]
                ta = np.array(d.amps, dtype=float)[list(op.dst)]
                tp = np.array(d.phases_rad, dtype=float)[list(op.dst)]

                # take current src values
                sf = start.freqs_hz[list(op.src)]
                sa = start.amps[list(op.src)]
                sp = start.phases_rad[list(op.src)]

                if len(sf) != len(tf):
                    raise ValueError(f"remap_from_def: src len {len(sf)} != dst len {len(tf)}")

                end = PlaneState(freqs_hz=tf, amps=ta, phases_rad=tp)

                dt = _round_to_samples(fs, op.time_s)
                if dt > 0:
                    for p in spec.planes:
                        if p == op.plane:
                            spans[p].append(
                                Span(t, t + dt, PlaneState(sf, sa, sp), end, interp=op.kind, seg_name=seg.name)
                            )
                        else:
                            st = cur[p]
                            spans[p].append(Span(t, t + dt, st, st, interp="hold", seg_name=seg.name))
                    t += dt
                    seg_time_accum += dt
                cur[op.plane] = end
                continue

            if isinstance(op, MoveOp):
                start = cur[op.plane]
                n = len(start.freqs_hz)
                idx = _select_idxs(n, op.idxs)

                f1 = start.freqs_hz.copy()
                f1[idx] = f1[idx] + float(op.df_hz)

                end = PlaneState(freqs_hz=f1, amps=start.amps.copy(), phases_rad=start.phases_rad.copy())

                dt = _round_to_samples(fs, op.time_s)
                if dt > 0:
                    for p in spec.planes:
                        if p == op.plane:
                            spans[p].append(Span(t, t + dt, start, end, interp=op.kind, seg_name=seg.name))
                        else:
                            st = cur[p]
                            spans[p].append(Span(t, t + dt, st, st, interp="hold", seg_name=seg.name))
                    t += dt
                    seg_time_accum += dt
                cur[op.plane] = end
                continue

            if isinstance(op, RampAmpToOp):
                start = cur[op.plane]
                n = len(start.amps)
                idx = _select_idxs(n, op.idxs)

                a1 = start.amps.copy()
                if isinstance(op.amps_target, tuple):
                    tgt = np.array(op.amps_target, dtype=float)
                    if len(tgt) == 1:
                        tgt = np.repeat(tgt, len(idx))
                    if len(tgt) != len(idx):
                        raise ValueError("ramp_amp_to: target length mismatch for selected idxs")
                    a1[idx] = tgt
                else:
                    a1[idx] = float(op.amps_target)

                end = PlaneState(freqs_hz=start.freqs_hz.copy(), amps=a1, phases_rad=start.phases_rad.copy())

                dt = _round_to_samples(fs, op.time_s)
                if dt > 0:
                    for p in spec.planes:
                        if p == op.plane:
                            spans[p].append(
                                Span(t, t + dt, start, end, interp=op.kind, tau_s=op.tau_s, seg_name=seg.name)
                            )
                        else:
                            st = cur[p]
                            spans[p].append(Span(t, t + dt, st, st, interp="hold", seg_name=seg.name))
                    t += dt
                    seg_time_accum += dt
                cur[op.plane] = end
                continue

            if isinstance(op, HoldOp):
                dt = _round_to_samples(fs, op.time_s)
                if dt <= 0:
                    # user wants "cursor doesn't move", but segments must be >= 1 sample overall.
                    # so a hold(0) is a state-noop; allowed.
                    continue

                # In wait_trig segments: snap frequencies to nearest wrap-continuous values
                if seg.mode == "wait_trig":
                    for p in spec.planes:
                        st = cur[p]
                        snapped = _snap_freqs_to_wrap(st.freqs_hz, dt)
                        df = np.max(np.abs(snapped - st.freqs_hz)) if st.freqs_hz.size else 0.0
                        if op.warn_df_hz is not None and df > op.warn_df_hz:
                            warnings.warn(
                                f"[{seg.name}] snapping on plane {p}: max |df|={df:.3g} Hz exceeds warn_df={op.warn_df_hz:.3g} Hz"
                            )
                        cur[p] = PlaneState(snapped, st.amps.copy(), st.phases_rad.copy())

                # Hold applies to all planes (continuous timeline)
                for p in spec.planes:
                    st = cur[p]
                    spans[p].append(Span(t, t + dt, st, st, interp="hold", seg_name=seg.name))
                t += dt
                seg_time_accum += dt
                continue

            raise TypeError(f"Unknown op type: {type(op)}")

        # Enforce: every segment must be >= 1 sample long
        min_seg = 1.0 / fs
        if seg_time_accum < min_seg:
            raise ValueError(
                f"Segment '{seg.name}' has duration {seg_time_accum:.3g}s < 1 sample ({min_seg:.3g}s). "
                "Add a hold() or a timed op so the segment is at least 1 sample long."
            )

    # Make sure each plane has at least one span, for plotting convenience
    for p in spec.planes:
        if not spans[p]:
            st = cur[p]
            spans[p].append(Span(0.0, 1.0 / fs, st, st, interp="hold", seg_name="__dummy__"))

    return ResolvedTimeline(
        sample_rate_hz=fs,
        planes=spans,
        segment_starts=segment_starts,
        t_end=t,
    )
