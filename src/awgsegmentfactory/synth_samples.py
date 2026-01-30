"""Sample synthesis: `QuantizedIR` → `CompiledSequenceProgram`.

This stage synthesizes per-segment, per-channel int16 samples (pattern memory) and
builds a simple step table (sequence memory) that chains segments in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from .resolved_ir import ResolvedLogicalChannelPart, ResolvedSegment
from .interpolation import interp_param
from .quantize import QuantizedIR, SegmentQuantizationInfo
from .types import ChannelMap


@dataclass(frozen=True)
class CompiledSegment:
    """One compiled hardware segment: an int16 array per hardware channel."""

    name: str
    n_samples: int
    # (n_channels, n_samples); NumPy by default. When `compile_sequence_program(..., output="cupy")`,
    # this may be a CuPy ndarray (device resident).
    data_i16: Any


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
    logical_channel_to_hardware_channel: ChannelMap
    gain: float
    clip: float
    full_scale: int
    segments: Tuple[CompiledSegment, ...]
    steps: Tuple[SequenceStep, ...]
    quantization: Tuple[SegmentQuantizationInfo, ...]


def _interp_logical_channel_part(
    pp: ResolvedLogicalChannelPart,
    *,
    n_samples: int,
    sample_rate_hz: float,
    xp: Any = np,
) -> tuple[Any, Any]:
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
        z = xp.zeros((n_samples, 0), dtype=float)
        return z, z

    if pp.interp.kind == "hold":
        dtype = xp.float32 if xp is not np else float
        freqs = xp.repeat(xp.asarray(f0, dtype=dtype)[None, :], n_samples, axis=0)
        amps = xp.repeat(xp.asarray(a0, dtype=dtype)[None, :], n_samples, axis=0)
        return freqs, amps

    dtype = xp.float32 if xp is not np else float
    u = (xp.arange(n_samples, dtype=dtype) / float(n_samples))[:, None]  # [0,1)
    t = (xp.arange(n_samples, dtype=dtype) / float(sample_rate_hz))[:, None]
    freqs = interp_param(
        xp.asarray(f0, dtype=dtype),
        xp.asarray(f1, dtype=dtype),
        interp=pp.interp,
        u=u,
        t_s=t,
        xp=xp,
    )
    amps = interp_param(
        xp.asarray(a0, dtype=dtype),
        xp.asarray(a1, dtype=dtype),
        interp=pp.interp,
        u=u,
        t_s=t,
        xp=xp,
    )
    return freqs, amps


def _synth_part(
    pp: ResolvedLogicalChannelPart,
    *,
    n_samples: int,
    sample_rate_hz: float,
    phase0_rad: Any,
    xp: Any = np,
) -> tuple[Any, Any]:
    """
    Synthesize one logical-channel part.

    Returns (y, phase_end) where:
    - y is (n_samples,) float waveform
    - phase_end is (n_tones,) phase at the end of the part (for carry)
    """
    freqs, amps = _interp_logical_channel_part(
        pp, n_samples=n_samples, sample_rate_hz=sample_rate_hz, xp=xp
    )
    if int(freqs.shape[1]) != int(phase0_rad.shape[0]):
        raise ValueError("phase0 length mismatch with tone count")

    if freqs.shape[1] == 0:
        return xp.zeros((n_samples,), dtype=float), phase0_rad.copy()

    dt = 1.0 / float(sample_rate_hz)
    dphi = freqs
    dphi *= float(2.0 * np.pi * dt)  # (n_samples, n_tones)
    phase_end = (phase0_rad + xp.sum(dphi, axis=0)) % (2.0 * np.pi)

    # phi[n] is the phase *at* sample n.
    phi = xp.cumsum(dphi, axis=0)
    phi += phase0_rad[None, :]
    phi -= dphi
    y = xp.sum(amps * xp.sin(phi), axis=1)
    return y, phase_end


def _synth_logical_channel_segment(
    seg: ResolvedSegment,
    *,
    logical_channel: str,
    sample_rate_hz: float,
    phase_in: Optional[Any],
    xp: Any = np,
) -> tuple[Any, Any]:
    """Synthesize one logical channel's waveform for a full segment (with optional phase carry)."""
    if not seg.parts:
        return xp.zeros((0,), dtype=float), xp.zeros((0,), dtype=float)

    first = seg.parts[0].logical_channels[logical_channel]
    n_tones = int(first.start.freqs_hz.shape[0])

    if seg.phase_mode == "carry":
        if phase_in is None:
            phase = xp.asarray(first.start.phases_rad).copy()
        else:
            # Policy: phases zip by index. Extra current tones start at their declared
            # start phase (defaults to 0). Extra previous phases are dropped.
            phase = xp.asarray(first.start.phases_rad).copy()
            n_copy = min(int(phase_in.shape[0]), n_tones)
            phase[:n_copy] = phase_in[:n_copy]
    else:
        phase = xp.asarray(first.start.phases_rad).copy()

    ys: list[Any] = []
    for part in seg.parts:
        pp = part.logical_channels[logical_channel]
        y, phase = _synth_part(
            pp,
            n_samples=part.n_samples,
            sample_rate_hz=sample_rate_hz,
            phase0_rad=phase,
            xp=xp,
        )
        ys.append(y)
    return xp.concatenate(ys, axis=0), phase


def _cupy_or_raise():
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gpu=True requires CuPy to be installed and importable."
        ) from exc

    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gpu=True but the CUDA runtime is not available (NVIDIA driver/CUDA mismatch?)."
        ) from exc
    if n <= 0:  # pragma: no cover
        raise RuntimeError("gpu=True but no NVIDIA CUDA devices were detected.")
    return cp


def compiled_sequence_program_to_numpy(
    prog: CompiledSequenceProgram,
) -> CompiledSequenceProgram:
    """
    Convert a compiled program to CPU/NumPy arrays.

    This is a no-op if segment buffers are already NumPy. If segment buffers are CuPy,
    this performs a device->host transfer for each segment buffer.
    """
    out_segments: list[CompiledSegment] = []
    for seg in prog.segments:
        data = seg.data_i16
        if isinstance(data, np.ndarray):
            out_segments.append(seg)
            continue
        try:
            import cupy as cp  # type: ignore

            data_np = cp.asnumpy(data)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Cannot convert compiled segment data to NumPy (CuPy not available?)."
            ) from exc
        out_segments.append(
            CompiledSegment(name=seg.name, n_samples=seg.n_samples, data_i16=data_np)
        )

    if all(a is b for a, b in zip(out_segments, prog.segments, strict=True)):
        return prog

    return CompiledSequenceProgram(
        sample_rate_hz=prog.sample_rate_hz,
        logical_channel_to_hardware_channel=dict(prog.logical_channel_to_hardware_channel),
        gain=float(prog.gain),
        clip=float(prog.clip),
        full_scale=int(prog.full_scale),
        segments=tuple(out_segments),
        steps=tuple(prog.steps),
        quantization=tuple(prog.quantization),
    )


def compile_sequence_program(
    quantized: QuantizedIR,
    *,
    gain: float,
    clip: float,
    full_scale: int,
    gpu: bool = False,
    output: Literal["numpy", "cupy"] = "numpy",
) -> CompiledSequenceProgram:
    """
    Compile a QuantizedIR into per-segment, per-channel int16 sample arrays suitable
    for writing into an AWG "sequence mode" pattern memory.

    GPU mode (`gpu=True`):
    - Uses CuPy for the inner synthesis math in `_synth_part`:
      interpolation, phase accumulation (`cumsum`), `sin`, and tone summation.
    - Applies gain/clip and int16 quantization on the GPU.
    - Output buffers:
      - `output="numpy"`: transfers the final `(n_channels, n_samples)` int16 segment buffer
        back to NumPy once per segment (device->host).
      - `output="cupy"`: keeps int16 buffers on the GPU as CuPy arrays (no device->host copy).

    Note: resolve/quantize stages are still CPU/NumPy.
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

    q_ir = quantized.resolved_ir
    logical_channel_to_hardware_channel = quantized.logical_channel_to_hardware_channel
    q_info = quantized.quantization

    n_segments = len(q_ir.segments)
    if n_segments == 0:
        raise ValueError("No segments to compile")

    n_channels = max(logical_channel_to_hardware_channel.values()) + 1

    compiled_segments: list[CompiledSegment] = []
    compiled_steps: list[SequenceStep] = []

    if output not in ("numpy", "cupy"):
        raise ValueError("output must be 'numpy' or 'cupy'")
    if output == "cupy" and not gpu:
        raise ValueError("output='cupy' requires gpu=True")

    xp: Any = np
    cp: Any = None
    if gpu:
        cp = _cupy_or_raise()
        xp = cp

    # Phase state carried across segments, per logical channel.
    phase_state: dict[str, Any] = {}

    for i, seg in enumerate(q_ir.segments):
        n = seg.n_samples
        data = xp.zeros((n_channels, n), dtype=xp.int16)

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
                xp=xp,
            )
            phase_state[logical_channel] = phase_out

            y = xp.clip(gain * y, -float(clip), float(clip))
            samp = xp.rint(y * float(full_scale)).astype(xp.int16)
            data[hw_ch, :] = samp

        if cp is not None and output == "numpy":
            data_i16 = cp.asnumpy(data)
        else:
            data_i16 = data

        compiled_segments.append(
            CompiledSegment(name=seg.name, n_samples=n, data_i16=data_i16)
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
