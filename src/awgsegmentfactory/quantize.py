"""Quantization and hardware-constrained transforms for sequence mode.

This stage takes a `ResolvedIR` and applies Spectrum sequence-mode constraints:
- segment length snapping (minimum sizes, step sizes, optional global quantum)
- optional wrap-friendly snapping for constant, loopable segments

The result is a `QuantizedIR` (a `ResolvedIR` plus quantization metadata and channel mapping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .resolved_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from .resolved_timeline import LogicalChannelState
from .types import ChannelMap
from .intent_ir import InterpSpec


@dataclass(frozen=True)
class SegmentQuantizationInfo:
    """Per-segment metadata describing how/why its length was quantized."""

    name: str
    mode: str
    loop: int
    loopable: bool
    original_samples: int
    quantized_samples: int
    step_samples: int
    quantum_samples: int
    sample_rate_hz: float

    @property
    def original_s(self) -> float:
        """Original segment duration in seconds (pre-quantization)."""
        return self.original_samples / self.sample_rate_hz

    @property
    def quantized_s(self) -> float:
        """Quantized segment duration in seconds (post-quantization)."""
        return self.quantized_samples / self.sample_rate_hz


@dataclass(frozen=True)
class QuantizedIR:
    """Output of `quantize_resolved_ir`: a resolved IR plus quantization metadata."""

    resolved_ir: ResolvedIR
    logical_channel_to_hardware_channel: ChannelMap
    quantization: tuple[SegmentQuantizationInfo, ...]

    @property
    def sample_rate_hz(self) -> float:
        """Sample rate in Hz (forwarded from the underlying resolved IR)."""
        return float(self.resolved_ir.sample_rate_hz)

    @property
    def logical_channels(self) -> tuple[str, ...]:
        """Ordered logical channel names present in the program."""
        return self.resolved_ir.logical_channels

    @property
    def segments(self) -> tuple[ResolvedSegment, ...]:
        """Quantized resolved segments (pattern-memory candidates)."""
        return self.resolved_ir.segments

    def to_timeline(self):
        """Convenience: forward to `ResolvedIR.to_timeline()` for debug plotting."""
        return self.resolved_ir.to_timeline()


def format_samples_time(
    n_samples: int, sample_rate_hz: float, *, unit: str = "us"
) -> str:
    """Format a duration as `N (X unit)` for post-quantization user display."""
    if unit == "us":
        scale = 1e6
        suffix = "µs"
    elif unit == "ms":
        scale = 1e3
        suffix = "ms"
    elif unit == "s":
        scale = 1.0
        suffix = "s"
    else:
        raise ValueError(f"Unknown unit {unit!r}")
    t = scale * (float(n_samples) / float(sample_rate_hz))
    return f"{n_samples} ({t:.4f} {suffix})"


def _ceil_to_multiple(n: int, m: int) -> int:
    """Ceil `n` up to the next multiple of `m` (return 0 for n<=0)."""
    if m <= 0:
        raise ValueError("m must be > 0")
    if n <= 0:
        return 0
    return int(((n + m - 1) // m) * m)


def _round_to_multiple(n: int, m: int) -> int:
    """Round `n` to the nearest multiple of `m` (return 0 for n<=0)."""
    if m <= 0:
        raise ValueError("m must be > 0")
    if n <= 0:
        return 0
    return int(round(n / m)) * m


def quantum_samples(
    sample_rate_hz: float, *, quantum_s: float, step_samples: int
) -> int:
    """
    Choose a global segment quantum (e.g. ~40us) in machine units (samples),
    rounded to the nearest hardware step size.
    """
    if quantum_s <= 0:
        raise ValueError("quantum_s must be > 0")
    if step_samples <= 0:
        raise ValueError("step_samples must be > 0")
    n = int(round(float(sample_rate_hz) * float(quantum_s)))
    q = _round_to_multiple(n, step_samples)
    return max(step_samples, q)


def min_segment_samples_per_channel(*, n_channels: int) -> int:
    """
    Spectrum sequence mode minimum segment sizes (samples per channel):
      1ch: 384
      2ch: 192
      4ch: 96
    """
    if n_channels not in (1, 2, 4):
        raise ValueError("n_channels must be 1, 2, or 4")
    return 384 // n_channels


def _segment_is_constant(seg: ResolvedSegment, logical_channel: str) -> bool:
    """
    True if a segment is a pure hold for the given logical channel: no parameter changes
    across time, so it can be repeated without discontinuities (aside from phase wrap).
    """
    if not seg.parts:
        return True
    ref: Optional[LogicalChannelState] = None
    for part in seg.parts:
        pp = part.logical_channels[logical_channel]
        if pp.interp.kind != "hold":
            return False
        if not np.array_equal(pp.start.freqs_hz, pp.end.freqs_hz):
            return False
        if not np.array_equal(pp.start.amps, pp.end.amps):
            return False
        if not np.array_equal(pp.start.phases_rad, pp.end.phases_rad):
            return False
        if ref is None:
            ref = pp.start
        else:
            if not np.array_equal(pp.start.freqs_hz, ref.freqs_hz):
                return False
            if not np.array_equal(pp.start.amps, ref.amps):
                return False
            if not np.array_equal(pp.start.phases_rad, ref.phases_rad):
                return False
    return True


def _snap_freqs_to_wrap(
    freqs_hz: np.ndarray, *, n_samples: int, sample_rate_hz: float
) -> np.ndarray:
    """Snap frequencies so `freq * segment_len` is an integer (phase-continuous on loop)."""
    if n_samples <= 0:
        return freqs_hz
    seg_len_s = float(n_samples) / float(sample_rate_hz)
    k = np.round(freqs_hz * seg_len_s)
    return k / seg_len_s


def quantize_resolved_ir(
    ir: ResolvedIR,
    *,
    logical_channel_to_hardware_channel: ChannelMap,
    segment_quantum_s: float = 40e-6,
    step_samples: int = 32,
) -> QuantizedIR:
    """
    Quantise segments for Spectrum sequence mode constraints.

    Rules:
    - All segments: length rounded up to a multiple of `step_samples` (and minimum size)
    - Loopable segments (wait_trig or loop>1): length rounded to nearest multiple of `quantum_samples`
      (but at least one quantum), and constant segments get frequency wrap-snapping.
    """
    fs = float(ir.sample_rate_hz)
    q_samples = quantum_samples(
        fs, quantum_s=segment_quantum_s, step_samples=step_samples
    )

    n_channels = len(set(logical_channel_to_hardware_channel.values()))
    min_samples = min_segment_samples_per_channel(n_channels=n_channels)
    min_samples = _ceil_to_multiple(min_samples, step_samples)

    infos: list[SegmentQuantizationInfo] = []
    out_segments: list[ResolvedSegment] = []

    for seg in ir.segments:
        loopable = seg.mode == "wait_trig" or seg.loop > 1
        n0 = int(seg.n_samples)

        constant = all(_segment_is_constant(seg, lc) for lc in ir.logical_channels)

        if loopable:
            # For loopable segments, prefer a global "quantum" length. For non-constant
            # segments we only ever round up (never truncate waveform content).
            if constant:
                n1 = _round_to_multiple(n0, q_samples)
            else:
                n1 = _ceil_to_multiple(n0, q_samples)
            if n1 < q_samples:
                n1 = q_samples
        else:
            n1 = _ceil_to_multiple(n0, step_samples)

        if n1 < min_samples:
            n1 = min_samples

        infos.append(
            SegmentQuantizationInfo(
                name=seg.name,
                mode=str(seg.mode),
                loop=int(seg.loop),
                loopable=loopable,
                original_samples=n0,
                quantized_samples=n1,
                step_samples=step_samples,
                quantum_samples=q_samples,
                sample_rate_hz=fs,
            )
        )

        parts: list[ResolvedPart] = list(seg.parts)
        n_parts = sum(p.n_samples for p in parts)
        if n_parts != n0:  # pragma: no cover
            raise RuntimeError("Segment parts/sample count mismatch")

        extra = n1 - n0
        if extra > 0:
            # Pad segment to `n1` by appending a hold part at the end.
            if not parts:
                raise RuntimeError(
                    "resolve_intent_ir produced a segment with no parts"
                )
            hold_logical_channels = {
                lc: ResolvedLogicalChannelPart(
                    start=parts[-1].logical_channels[lc].end,
                    end=parts[-1].logical_channels[lc].end,
                    interp=InterpSpec("hold"),
                )
                for lc in ir.logical_channels
            }
            parts.append(
                ResolvedPart(n_samples=extra, logical_channels=hold_logical_channels)
            )
        elif extra < 0:
            # Shorten only constant loopable segments (safe to truncate holds).
            if not constant:
                raise ValueError(
                    f"Segment {seg.name!r} is loopable but not constant; refusing to shorten "
                    f"from {n0} to {n1} samples. Make it constant or increase duration."
                )
            keep = n1
            trimmed: list[ResolvedPart] = []
            for part in parts:
                if keep <= 0:
                    break
                if part.n_samples <= keep:
                    trimmed.append(part)
                    keep -= part.n_samples
                else:
                    trimmed.append(
                        ResolvedPart(n_samples=keep, logical_channels=part.logical_channels)
                    )
                    keep = 0
            parts = trimmed
            if sum(p.n_samples for p in parts) != n1:  # pragma: no cover
                raise RuntimeError("Failed to trim segment parts correctly")

        # For loopable constant segments, snap freqs to wrap for the final length.
        if loopable and constant:
            seg_len = n1
            new_parts: list[ResolvedPart] = []
            for part in parts:
                new_logical_channels: dict[str, ResolvedLogicalChannelPart] = {}
                for lc in ir.logical_channels:
                    pp = part.logical_channels[lc]
                    snapped = _snap_freqs_to_wrap(
                        pp.start.freqs_hz, n_samples=seg_len, sample_rate_hz=fs
                    )
                    st = LogicalChannelState(
                        snapped, pp.start.amps.copy(), pp.start.phases_rad.copy()
                    )
                    new_logical_channels[lc] = ResolvedLogicalChannelPart(
                        start=st, end=st, interp=InterpSpec("hold")
                    )
                new_parts.append(
                    ResolvedPart(
                        n_samples=part.n_samples,
                        logical_channels=new_logical_channels,
                    )
                )
            parts = new_parts

        out_segments.append(
            ResolvedSegment(
                name=seg.name,
                mode=seg.mode,
                loop=seg.loop,
                parts=tuple(parts),
                phase_mode=seg.phase_mode,
            )
        )

    q_ir = ResolvedIR(
        sample_rate_hz=fs,
        logical_channels=ir.logical_channels,
        segments=tuple(out_segments),
    )
    return QuantizedIR(
        resolved_ir=q_ir,
        logical_channel_to_hardware_channel=dict(logical_channel_to_hardware_channel),
        quantization=tuple(infos),
    )
