"""Resolver: `IntentIR` → `ResolvedIR` (discretize into integer-sample primitives).

This stage turns continuous-time operations (`time_s`) into a segment/part structure
with integer `n_samples` suitable for later quantization and sample synthesis.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .ir import (
    IntentIR,
    HoldOp,
    UseDefOp,
    MoveOp,
    RampAmpToOp,
    RemapFromDefOp,
)
from .program_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from .timeline import LogicalChannelState


def _empty_state() -> LogicalChannelState:
    """Return a logical-channel state with zero tones (used as resolver initial state)."""
    return LogicalChannelState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


def _ceil_samples(sample_rate_hz: float, time_s: float) -> int:
    """Convert seconds to integer samples by ceiling (never truncates timed ops)."""
    if time_s <= 0:
        return 0
    return int(np.ceil(float(time_s) * float(sample_rate_hz)))


def _select_idxs(n: int, idxs: Optional[Tuple[int, ...]]) -> np.ndarray:
    """Validate and normalize an optional index list into a NumPy index array."""
    if idxs is None:
        return np.arange(n, dtype=int)
    idx = np.array(list(idxs), dtype=int)
    if np.any(idx < 0) or np.any(idx >= n):
        raise IndexError(f"idxs out of range for n={n}: {idxs}")
    return idx


def _hold_parts(
    intent: IntentIR, cur: Dict[str, LogicalChannelState]
) -> Dict[str, ResolvedLogicalChannelPart]:
    """Create per-logical-channel "hold" parts representing the current state."""
    return {
        lc: ResolvedLogicalChannelPart(start=cur[lc], end=cur[lc], interp="hold")
        for lc in intent.logical_channels
    }


def _append_target_part(
    parts: List[ResolvedPart],
    *,
    intent: IntentIR,
    cur: Dict[str, LogicalChannelState],
    n_samples: int,
    logical_channel: str,
    target_part: ResolvedLogicalChannelPart,
) -> None:
    """Append one `ResolvedPart` that changes only `logical_channel`, holding the rest."""
    if n_samples <= 0:
        return
    logical_channels = _hold_parts(intent, cur)
    logical_channels[logical_channel] = target_part
    parts.append(ResolvedPart(n_samples=n_samples, logical_channels=logical_channels))


def resolve_intent_ir(intent: IntentIR, *, sample_rate_hz: float) -> ResolvedIR:
    """
    Resolve an IntentIR into a fully-explicit ResolvedIR (integer-sample primitives).

    Guarantees:
    - state is carried across segment boundaries (no implicit resets)
    - every timed op produces a ResolvedPart containing *all* logical channels
    - `time=0` ops update state without advancing time
    """
    fs = float(sample_rate_hz)
    if fs <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    # current per-logical-channel state
    cur: Dict[str, LogicalChannelState] = {
        lc: _empty_state() for lc in intent.logical_channels
    }
    segments: List[ResolvedSegment] = []

    for seg in intent.segments:
        parts: List[ResolvedPart] = []
        seg_samples = 0

        for op in seg.ops:
            if isinstance(op, UseDefOp):
                d = intent.definitions[op.def_name]
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
                d = intent.definitions[op.target_def]
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
                        intent=intent,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=ResolvedLogicalChannelPart(
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
                        intent=intent,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=ResolvedLogicalChannelPart(
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
                        intent=intent,
                        cur=cur,
                        n_samples=n,
                        logical_channel=op.logical_channel,
                        target_part=ResolvedLogicalChannelPart(
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
                    ResolvedPart(n_samples=n, logical_channels=_hold_parts(intent, cur))
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
            ResolvedSegment(
                name=seg.name,
                mode=seg.mode,
                loop=seg.loop,
                parts=tuple(parts),
                phase_mode=seg.phase_mode,
            )
        )

    return ResolvedIR(
        sample_rate_hz=fs,
        logical_channels=intent.logical_channels,
        segments=tuple(segments),
    )
