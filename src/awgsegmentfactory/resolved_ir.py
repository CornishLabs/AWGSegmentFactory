"""Resolved IR dataclasses (segment/part primitives in integer samples).

`ResolvedIR` is the main compiler input: it groups a program into segments, each
containing parts with integer `n_samples` and per-logical-channel interpolation specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .intent_ir import InterpSpec, SegmentMode, SegmentPhaseMode
from .resolved_timeline import LogicalChannelState, ResolvedTimeline, Span


@dataclass(frozen=True)
class ResolvedLogicalChannelPart:
    """Per-logical-channel primitive for a single time interval (part)."""

    start: LogicalChannelState
    end: LogicalChannelState
    interp: InterpSpec


@dataclass(frozen=True)
class ResolvedPart:
    """
    A time interval with a fixed interpolation primitive per logical channel.

    The compiler-friendly representation is:
    - duration is stored as `n_samples` (integer)
    - each logical channel has (start, end, interp) describing parameter evolution
    """

    n_samples: int
    logical_channels: Dict[str, ResolvedLogicalChannelPart]


@dataclass(frozen=True)
class ResolvedSegment:
    """
    A resolved segment: a list of integer-sample parts plus sequence metadata.

    `phase_mode` is carried through resolve/quantize, but is applied during sample
    synthesis (`QIRtoSamplesSegmentCompiler.compile_to_card_int16(...)` /
    `.compile_to_voltage_mV(...)`). The debug timeline view
    (`ResolvedIR.to_timeline()`) shows the pre-optimised phases stored in the IR.
    """

    name: str
    mode: SegmentMode
    loop: int
    parts: Tuple[ResolvedPart, ...]
    phase_mode: SegmentPhaseMode = "continue"
    # Quantization preferences (applied in `quantize_resolved_ir`).
    snap_len_to_quantum: bool = True
    snap_freqs_to_wrap: bool = True

    @property
    def n_samples(self) -> int:
        """Total segment length in samples (sum of `parts`)."""
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
        """Total program length in samples (sum of segments)."""
        return sum(s.n_samples for s in self.segments)

    @property
    def duration_s(self) -> float:
        """Total program duration in seconds (based on `sample_rate_hz`)."""
        return self.n_samples / self.sample_rate_hz

    def to_timeline(self) -> ResolvedTimeline:
        """Convert into a debug-friendly `ResolvedTimeline` of per-channel `Span`s."""
        spans_by_logical_channel: Dict[str, list[Span]] = {
            lc: [] for lc in self.logical_channels
        }
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
                    spans_by_logical_channel[lc].append(
                        Span(
                            t0=t0,
                            t1=t1,
                            start=pp.start,
                            end=pp.end,
                            interp=pp.interp,
                            seg_name=seg.name,
                        )
                    )
                n0 += part.n_samples

        return ResolvedTimeline(
            sample_rate_hz=fs,
            spans_by_logical_channel=spans_by_logical_channel,
            segment_starts=segment_starts,
            t_end=n0 / fs,
        )
