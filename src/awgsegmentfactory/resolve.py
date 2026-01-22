from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .ir import (
    ProgramSpec, SegmentSpec, SegmentMode,
    HoldOp, UseDefOp, MoveOp, RampAmpToOp, RemapFromDefOp,
)
from .program_ir import ProgramIR, SegmentIR, PartIR, PlanePartIR
from .timeline import PlaneState, ResolvedTimeline

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

def _ceil_samples(sample_rate_hz: float, time_s: float) -> int:
    if time_s <= 0:
        return 0
    return int(np.ceil(float(time_s) * float(sample_rate_hz)))

def _select_idxs(n: int, idxs: Optional[Tuple[int, ...]]) -> np.ndarray:
    if idxs is None:
        return np.arange(n, dtype=int)
    idx = np.array(list(idxs), dtype=int)
    if np.any(idx < 0) or np.any(idx >= n):
        raise IndexError(f"idxs out of range for n={n}: {idxs}")
    return idx

def resolve_program_ir(spec: ProgramSpec) -> ProgramIR:
    fs = spec.sample_rate_hz

    # current per-plane state
    cur: Dict[str, PlaneState] = {p: _empty_state() for p in spec.planes}
    segments: List[SegmentIR] = []

    for seg in spec.segments:
        parts: List[PartIR] = []
        seg_samples = 0

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

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    planes: Dict[str, PlanePartIR] = {}
                    for p in spec.planes:
                        if p == op.plane:
                            planes[p] = PlanePartIR(
                                start=PlaneState(sf, sa, sp),
                                end=end,
                                interp=op.kind,
                            )
                        else:
                            st = cur[p]
                            planes[p] = PlanePartIR(start=st, end=st, interp="hold")
                    parts.append(PartIR(n_samples=n, planes=planes))
                    seg_samples += n
                cur[op.plane] = end
                continue

            if isinstance(op, MoveOp):
                start = cur[op.plane]
                n = len(start.freqs_hz)
                idx = _select_idxs(n, op.idxs)

                f1 = start.freqs_hz.copy()
                f1[idx] = f1[idx] + float(op.df_hz)

                end = PlaneState(freqs_hz=f1, amps=start.amps.copy(), phases_rad=start.phases_rad.copy())

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    planes: Dict[str, PlanePartIR] = {}
                    for p in spec.planes:
                        if p == op.plane:
                            planes[p] = PlanePartIR(start=start, end=end, interp=op.kind)
                        else:
                            st = cur[p]
                            planes[p] = PlanePartIR(start=st, end=st, interp="hold")
                    parts.append(PartIR(n_samples=n, planes=planes))
                    seg_samples += n
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

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    planes: Dict[str, PlanePartIR] = {}
                    for p in spec.planes:
                        if p == op.plane:
                            planes[p] = PlanePartIR(start=start, end=end, interp=op.kind, tau_s=op.tau_s)
                        else:
                            st = cur[p]
                            planes[p] = PlanePartIR(start=st, end=st, interp="hold")
                    parts.append(PartIR(n_samples=n, planes=planes))
                    seg_samples += n
                cur[op.plane] = end
                continue

            if isinstance(op, HoldOp):
                n = _ceil_samples(fs, op.time_s)
                if n <= 0:
                    # user wants "cursor doesn't move", but segments must be >= 1 sample overall.
                    # so a hold(0) is a state-noop; allowed.
                    continue

                # Hold applies to all planes (continuous timeline)
                planes = {p: PlanePartIR(start=cur[p], end=cur[p], interp="hold") for p in spec.planes}
                parts.append(PartIR(n_samples=n, planes=planes))
                seg_samples += n
                continue

            raise TypeError(f"Unknown op type: {type(op)}")

        # Enforce: every segment must be >= 1 sample long
        if seg_samples < 1:
            min_seg = 1.0 / fs
            raise ValueError(
                f"Segment '{seg.name}' has duration {seg_samples / fs:.3g}s < 1 sample ({min_seg:.3g}s). "
                "Add a hold() or a timed op so the segment is at least 1 sample long."
            )

        segments.append(SegmentIR(name=seg.name, mode=seg.mode, loop=seg.loop, parts=tuple(parts), phase_mode=seg.phase_mode))

    return ProgramIR(
        sample_rate_hz=fs,
        planes=spec.planes,
        segments=tuple(segments),
    )


def resolve_program(spec: ProgramSpec) -> ResolvedTimeline:
    return resolve_program_ir(spec).to_timeline()
