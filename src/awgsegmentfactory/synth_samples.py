"""Sample synthesis: `QuantizedIR` → `CompiledSequenceProgram`.

This stage synthesizes per-segment, per-channel int16 samples (pattern memory) and
builds a simple step table (sequence memory) that chains segments in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .resolved_ir import ResolvedLogicalChannelPart, ResolvedSegment
from .interpolation import interp_param
from .quantize import QuantizedIR, SegmentQuantizationInfo


@dataclass(frozen=True)
class CompiledSegment:
    """One compiled hardware segment: an int16 array per hardware channel."""

    name: str
    n_samples: int
    data_i16: np.ndarray  # (n_channels, n_samples)


@dataclass(frozen=True)
class SequenceStep:
    """One sequence-step entry mapping to Spectrum "step memory" fields."""

    step_index: int
    segment_index: int
    next_step: int
    loops: int
    on_trig: bool


@dataclass(frozen=True)
class CompiledSequenceProgram:
    """Final compiler output: segments (pattern memory) plus steps (sequence memory)."""

    sample_rate_hz: float
    logical_channel_to_hardware_channel: Dict[str, int]
    gain: float
    clip: float
    full_scale: int
    segments: Tuple[CompiledSegment, ...]
    steps: Tuple[SequenceStep, ...]
    quantization: Tuple[SegmentQuantizationInfo, ...]


def _interp_logical_channel_part(
    pp: ResolvedLogicalChannelPart, *, n_samples: int, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (freqs_hz, amps) as (n_samples, n_tones) arrays.
    """
    f0 = pp.start.freqs_hz
    f1 = pp.end.freqs_hz
    a0 = pp.start.amps
    a1 = pp.end.amps

    if f0.shape != f1.shape or a0.shape != a1.shape:
        raise ValueError("Start/end shape mismatch for logical channel part")

    n_tones = int(f0.shape[0])
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    if n_tones == 0:
        z = np.zeros((n_samples, 0), dtype=float)
        return z, z

    if pp.interp == "hold":
        freqs = np.repeat(f0[None, :], n_samples, axis=0)
        amps = np.repeat(a0[None, :], n_samples, axis=0)
        return freqs, amps

    u = (np.arange(n_samples, dtype=float) / float(n_samples))[:, None]  # [0,1)
    t = (np.arange(n_samples, dtype=float) / float(sample_rate_hz))[:, None]
    freqs = interp_param(
        f0, f1, kind=pp.interp, u=u, t_s=t, tau_s=pp.tau_s
    )
    amps = interp_param(
        a0, a1, kind=pp.interp, u=u, t_s=t, tau_s=pp.tau_s
    )
    return freqs, amps


def _synth_part(
    pp: ResolvedLogicalChannelPart,
    *,
    n_samples: int,
    sample_rate_hz: float,
    phase0_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize one logical-channel part.

    Returns (y, phase_end) where:
    - y is (n_samples,) float waveform
    - phase_end is (n_tones,) phase at the end of the part (for carry)
    """
    freqs, amps = _interp_logical_channel_part(
        pp, n_samples=n_samples, sample_rate_hz=sample_rate_hz
    )
    if freqs.shape[1] != phase0_rad.shape[0]:
        raise ValueError("phase0 length mismatch with tone count")

    if freqs.shape[1] == 0:
        return np.zeros((n_samples,), dtype=float), phase0_rad.copy()

    dt = 1.0 / float(sample_rate_hz)
    dphi = 2.0 * np.pi * freqs * dt  # (n_samples, n_tones)

    # phi[n] is the phase *at* sample n.
    phi = phase0_rad[None, :] + np.cumsum(dphi, axis=0) - dphi
    y = np.sum(amps * np.sin(phi), axis=1)

    phase_end = (phase0_rad + np.sum(dphi, axis=0)) % (2.0 * np.pi)
    return y, phase_end


def _synth_logical_channel_segment(
    seg: ResolvedSegment,
    *,
    logical_channel: str,
    sample_rate_hz: float,
    phase_in: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize one logical channel's waveform for a full segment (with optional phase carry)."""
    if not seg.parts:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    first = seg.parts[0].logical_channels[logical_channel]
    n_tones = int(first.start.freqs_hz.shape[0])

    if seg.phase_mode == "carry":
        if phase_in is None:
            phase = first.start.phases_rad.copy()
        elif phase_in.shape != (n_tones,):
            raise ValueError(
                f"Cannot carry phase into segment {seg.name!r} logical_channel {logical_channel!r}: "
                f"tone count changed from {phase_in.shape[0]} to {n_tones}. "
                "Use phase_mode='fixed' for this segment or keep tone counts constant."
            )
        else:
            phase = phase_in.copy()
    else:
        phase = first.start.phases_rad.copy()

    ys: list[np.ndarray] = []
    for part in seg.parts:
        pp = part.logical_channels[logical_channel]
        y, phase = _synth_part(
            pp,
            n_samples=part.n_samples,
            sample_rate_hz=sample_rate_hz,
            phase0_rad=phase,
        )
        ys.append(y)
    return np.concatenate(ys, axis=0), phase


def compile_sequence_program(
    quantized: QuantizedIR,
    *,
    gain: float,
    clip: float,
    full_scale: int,
) -> CompiledSequenceProgram:
    """
    Compile a QuantizedIR into per-segment, per-channel int16 sample arrays suitable
    for writing into an AWG "sequence mode" pattern memory.
    """
    if gain <= 0:
        raise ValueError("gain must be > 0")
    if not (0 < clip <= 1.0):
        raise ValueError("clip must be in (0, 1]")
    if full_scale <= 0:
        raise ValueError("full_scale must be > 0")
    max_i16 = int(np.iinfo(np.int16).max)
    if full_scale > max_i16:
        raise ValueError(f"full_scale must be <= {max_i16} for int16 output")

    q_ir = quantized.ir
    logical_channel_to_hardware_channel = quantized.logical_channel_to_hardware_channel
    q_info = quantized.quantization

    n_segments = len(q_ir.segments)
    if n_segments == 0:
        raise ValueError("No segments to compile")

    n_channels = max(logical_channel_to_hardware_channel.values()) + 1

    compiled_segments: list[CompiledSegment] = []
    compiled_steps: list[SequenceStep] = []

    # Phase state carried across segments, per logical channel.
    phase_state: Dict[str, np.ndarray] = {}

    for i, seg in enumerate(q_ir.segments):
        n = seg.n_samples
        data = np.zeros((n_channels, n), dtype=np.int16)

        for logical_channel in q_ir.logical_channels:
            if logical_channel not in logical_channel_to_hardware_channel:
                raise KeyError(
                    f"No hardware channel mapping for logical_channel {logical_channel!r}"
                )
            hw_ch = int(logical_channel_to_hardware_channel[logical_channel])
            y, phase_out = _synth_logical_channel_segment(
                seg,
                logical_channel=logical_channel,
                sample_rate_hz=q_ir.sample_rate_hz,
                phase_in=phase_state.get(logical_channel),
            )
            phase_state[logical_channel] = phase_out

            y = gain * y
            y = np.clip(y, -1.0, 1.0)
            samp = np.round(y * (clip * float(full_scale))).astype(np.int16)
            data[hw_ch, :] = samp

        compiled_segments.append(
            CompiledSegment(name=seg.name, n_samples=n, data_i16=data)
        )

        on_trig = seg.mode == "wait_trig"
        loops = int(seg.loop) if seg.mode == "loop_n" else 1
        compiled_steps.append(
            SequenceStep(
                step_index=i,
                segment_index=i,
                next_step=(i + 1) % n_segments,
                loops=loops,
                on_trig=on_trig,
            )
        )

    return CompiledSequenceProgram(
        sample_rate_hz=q_ir.sample_rate_hz,
        logical_channel_to_hardware_channel=dict(logical_channel_to_hardware_channel),
        gain=float(gain),
        clip=float(clip),
        full_scale=int(full_scale),
        segments=tuple(compiled_segments),
        steps=tuple(compiled_steps),
        quantization=tuple(q_info),
    )
