from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .ir import InterpKind, SegmentMode, SegmentPhaseMode
from .timeline import LogicalChannelState, ResolvedTimeline, Span


@dataclass(frozen=True)
class ResolvedLogicalChannelPart:
    """Per-logical-channel primitive for a single time interval (part)."""

    start: LogicalChannelState
    end: LogicalChannelState
    interp: InterpKind
    tau_s: Optional[float] = None


@dataclass(frozen=True)
class ResolvedPart:
    """
    A time interval with a fixed interpolation primitive per logical channel.

    The compiler-friendly representation is:
    - duration is stored as `n_samples` (integer)
    - each logical channel has (start, end, interp, tau) describing parameter evolution
    """

    n_samples: int
    logical_channels: Dict[str, ResolvedLogicalChannelPart]


@dataclass(frozen=True)
class ResolvedSegment:
    name: str
    mode: SegmentMode
    loop: int
    parts: Tuple[ResolvedPart, ...]
    phase_mode: SegmentPhaseMode = "carry"

    @property
    def n_samples(self) -> int:
        return sum(p.n_samples for p in self.parts)


@dataclass(frozen=True)
class ResolvedIR:
    """
    Resolved, segment-grouped IR intended for compilation into AWG sequence mode.

    - `segments[i]` corresponds to a Spectrum "data segment" (pattern memory)
    - `mode/loop` map onto Spectrum sequence step settings
    - Each segment consists of primitive "parts" with integer sample lengths
    """

    sample_rate_hz: float
    logical_channels: Tuple[str, ...]
    segments: Tuple[ResolvedSegment, ...]

    @property
    def n_samples(self) -> int:
        return sum(s.n_samples for s in self.segments)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz

    def to_timeline(self) -> ResolvedTimeline:
        spans: Dict[str, list[Span]] = {lc: [] for lc in self.logical_channels}
        segment_starts: list[tuple[float, str]] = []

        fs = float(self.sample_rate_hz)
        n0 = 0
        for seg in self.segments:
            segment_starts.append((n0 / fs, seg.name))
            for part in seg.parts:
                t0 = n0 / fs
                t1 = (n0 + part.n_samples) / fs
                for lc in self.logical_channels:
                    pp = part.logical_channels[lc]
                    spans[lc].append(
                        Span(
                            t0=t0,
                            t1=t1,
                            start=pp.start,
                            end=pp.end,
                            interp=pp.interp,
                            tau_s=pp.tau_s,
                            seg_name=seg.name,
                        )
                    )
                n0 += part.n_samples

        return ResolvedTimeline(
            sample_rate_hz=fs,
            logical_channels=spans,
            segment_starts=segment_starts,
            t_end=n0 / fs,
        )
