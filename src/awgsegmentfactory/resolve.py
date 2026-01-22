from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .ir import (
    ProgramSpec,
    HoldOp,
    UseDefOp,
    MoveOp,
    RampAmpToOp,
    RemapFromDefOp,
)
from .program_ir import ProgramIR, SegmentIR, PartIR, LogicalChannelPartIR
from .timeline import LogicalChannelState, ResolvedTimeline


def _empty_state() -> LogicalChannelState:
    return LogicalChannelState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


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


def _hold_parts(
    spec: ProgramSpec, cur: Dict[str, LogicalChannelState]
) -> Dict[str, LogicalChannelPartIR]:
    return {
        lc: LogicalChannelPartIR(start=cur[lc], end=cur[lc], interp="hold")
        for lc in spec.logical_channels
    }


def _append_target_part(
    parts: List[PartIR],
    *,
    spec: ProgramSpec,
    cur: Dict[str, LogicalChannelState],
    n_samples: int,
    logical_channel: str,
    target_part: LogicalChannelPartIR,
) -> None:
    if n_samples <= 0:
        return
    logical_channels = _hold_parts(spec, cur)
    logical_channels[logical_channel] = target_part
    parts.append(PartIR(n_samples=n_samples, logical_channels=logical_channels))


def resolve_program_ir(spec: ProgramSpec) -> ProgramIR:
    """
    Resolve a ProgramSpec into a fully-explicit ProgramIR.

    Guarantees:
    - state is carried across segment boundaries (no implicit resets)
    - every timed op produces a PartIR containing *all* logical channels
    - `time=0` ops update state without advancing time
    """
    fs = spec.sample_rate_hz

    # current per-logical-channel state
    cur: Dict[str, LogicalChannelState] = {
        lc: _empty_state() for lc in spec.logical_channels
    }
    segments: List[SegmentIR] = []

    for seg in spec.segments:
        parts: List[PartIR] = []
        seg_samples = 0

        for op in seg.ops:
            if isinstance(op, UseDefOp):
                d = spec.definitions[op.def_name]
                if d.logical_channel != op.logical_channel:
                    raise ValueError(
                        f"Definition {d.name} is for logical_channel {d.logical_channel}, "
                        f"not {op.logical_channel}"
                    )

                cur[op.logical_channel] = LogicalChannelState(
                    freqs_hz=np.array(d.freqs_hz, dtype=float),
                    amps=np.array(d.amps, dtype=float),
                    phases_rad=np.array(d.phases_rad, dtype=float),
                )
                continue

            if isinstance(op, RemapFromDefOp):
                d = spec.definitions[op.target_def]
                if d.logical_channel != op.logical_channel:
                    raise ValueError(
                        f"Definition {d.name} is for logical_channel {d.logical_channel}, "
                        f"not {op.logical_channel}"
                    )

                start = cur[op.logical_channel]
                # build target arrays at dst indices
                tf = np.array(d.freqs_hz, dtype=float)[list(op.dst)]
                ta = np.array(d.amps, dtype=float)[list(op.dst)]
                tp = np.array(d.phases_rad, dtype=float)[list(op.dst)]

                # take current src values
                sf = start.freqs_hz[list(op.src)]
                sa = start.amps[list(op.src)]
                sp = start.phases_rad[list(op.src)]

                if len(sf) != len(tf):
                    raise ValueError(
                        f"remap_from_def: src len {len(sf)} != dst len {len(tf)}"
                    )

                end = LogicalChannelState(freqs_hz=tf, amps=ta, phases_rad=tp)

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    _append_target_part(
                        parts,
                        spec=spec,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=LogicalChannelPartIR(
                            start=LogicalChannelState(sf, sa, sp),
                            end=end,
                            interp=op.kind,
                        ),
                    )
                    seg_samples += n
                cur[op.logical_channel] = end
                continue

            if isinstance(op, MoveOp):
                start = cur[op.logical_channel]
                n = len(start.freqs_hz)
                idx = _select_idxs(n, op.idxs)

                f1 = start.freqs_hz.copy()
                f1[idx] = f1[idx] + float(op.df_hz)

                end = LogicalChannelState(
                    freqs_hz=f1,
                    amps=start.amps.copy(),
                    phases_rad=start.phases_rad.copy(),
                )

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    _append_target_part(
                        parts,
                        spec=spec,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=LogicalChannelPartIR(
                            start=start, end=end, interp=op.kind
                        ),
                    )
                    seg_samples += n
                cur[op.logical_channel] = end
                continue

            if isinstance(op, RampAmpToOp):
                start = cur[op.logical_channel]
                n = len(start.amps)
                idx = _select_idxs(n, op.idxs)

                a1 = start.amps.copy()
                if isinstance(op.amps_target, tuple):
                    tgt = np.array(op.amps_target, dtype=float)
                    if len(tgt) == 1:
                        tgt = np.repeat(tgt, len(idx))
                    if len(tgt) != len(idx):
                        raise ValueError(
                            "ramp_amp_to: target length mismatch for selected idxs"
                        )
                    a1[idx] = tgt
                else:
                    a1[idx] = float(op.amps_target)

                end = LogicalChannelState(
                    freqs_hz=start.freqs_hz.copy(),
                    amps=a1,
                    phases_rad=start.phases_rad.copy(),
                )

                n = _ceil_samples(fs, op.time_s)
                if n > 0:
                    _append_target_part(
                        parts,
                        spec=spec,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=LogicalChannelPartIR(
                            start=start,
                            end=end,
                            interp=op.kind,
                            tau_s=op.tau_s,
                        ),
                    )
                    seg_samples += n
                cur[op.logical_channel] = end
                continue

            if isinstance(op, HoldOp):
                n = _ceil_samples(fs, op.time_s)
                if n <= 0:
                    # user wants "cursor doesn't move", but segments must be >= 1 sample overall.
                    # so a hold(0) is a state-noop; allowed.
                    continue

                # Hold applies to all logical channels (continuous timeline)
                parts.append(
                    PartIR(n_samples=n, logical_channels=_hold_parts(spec, cur))
                )
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

        segments.append(
            SegmentIR(
                name=seg.name,
                mode=seg.mode,
                loop=seg.loop,
                parts=tuple(parts),
                phase_mode=seg.phase_mode,
            )
        )

    return ProgramIR(
        sample_rate_hz=fs,
        logical_channels=spec.logical_channels,
        segments=tuple(segments),
    )


def resolve_program(spec: ProgramSpec) -> ResolvedTimeline:
    return resolve_program_ir(spec).to_timeline()
