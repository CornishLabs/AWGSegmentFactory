"""Sample synthesis and sample quantization.

Pipeline in this module:
- `synthesize_sequence_program`: `QuantizedIR` -> float waveform buffers
- `quantize_synthesized_program`: float buffers -> int16 buffers
- `compile_sequence_program`: convenience wrapper for both stages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from .calibration import AWGPhysicalSetupInfo, OpticalPowerToRFAmpCalib
from .resolved_ir import ResolvedLogicalChannelPart, ResolvedSegment
from .interpolation import interp_param
from .phase_minimiser import minimise_crest_factor_phases
from .quantize import QuantizedIR, SegmentQuantizationInfo


@dataclass(frozen=True)
class SynthesizedSegment:
    """One synthesized hardware segment: float waveform per hardware channel."""

    name: str
    n_samples: int
    # (n_channels, n_samples); NumPy by default. When `synthesize_sequence_program(..., output="cupy")`,
    # this may be a CuPy ndarray (device resident).
    data: Any


@dataclass(frozen=True)
class CompiledSegment:
    """One quantized hardware segment: an int16 array per hardware channel."""

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
class SynthesizedSequenceProgram:
    """Float synthesis output: per-segment waveforms and sequence table."""

    sample_rate_hz: float
    physical_setup: AWGPhysicalSetupInfo
    segments: Tuple[SynthesizedSegment, ...]
    steps: Tuple[SequenceStep, ...]
    quantization: Tuple[SegmentQuantizationInfo, ...]


@dataclass(frozen=True)
class CompiledSequenceProgram:
    """Final compiler output: int16 segments (pattern memory) plus steps."""

    sample_rate_hz: float
    physical_setup: AWGPhysicalSetupInfo
    gain: float
    clip: float
    full_scale: int
    segments: Tuple[CompiledSegment, ...]
    steps: Tuple[SequenceStep, ...]
    quantization: Tuple[SegmentQuantizationInfo, ...]


@dataclass(frozen=True)
class _PhaseContinueState:
    """Per-logical-channel state needed to continue phases across segments."""

    freqs_hz: np.ndarray  # (n_tones,) end-of-segment freqs
    phases_rad: np.ndarray  # (n_tones,) end-of-segment phases


def _match_tones_by_frequency(
    prev_freqs_hz: np.ndarray,
    cur_freqs_hz: np.ndarray,
    *,
    tol_hz: float = 1.0,
) -> dict[int, int]:
    """
    Return a mapping {cur_idx -> prev_idx} by greedy nearest-frequency matching.

    The intent is to continue phases across segments even if tone order changes.
    """
    prev = np.asarray(prev_freqs_hz, dtype=float).reshape(-1)
    cur = np.asarray(cur_freqs_hz, dtype=float).reshape(-1)
    if prev.size == 0 or cur.size == 0:
        return {}

    # Candidate pairs within tolerance.
    d = np.abs(prev[:, None] - cur[None, :])
    pairs = np.argwhere(d <= float(tol_hz))
    if pairs.size == 0:
        return {}

    # Sort by smallest frequency difference.
    diffs = d[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(diffs, kind="stable")

    used_prev: set[int] = set()
    used_cur: set[int] = set()
    out: dict[int, int] = {}
    for idx in order:
        i_prev = int(pairs[idx, 0])
        i_cur = int(pairs[idx, 1])
        if i_prev in used_prev or i_cur in used_cur:
            continue
        used_prev.add(i_prev)
        used_cur.add(i_cur)
        out[i_cur] = i_prev
    return out


def _default_crest_time_grid_s(
    freqs_hz: np.ndarray,
    *,
    samples_per_period: int = 8,
    min_duration_s: float = 0.5e-6,
    max_duration_s: float = 5.0e-6,
) -> np.ndarray:
    """Pick a deterministic time grid for crest-factor optimisation."""
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if f.size == 0:
        return np.zeros((1,), dtype=float)

    f_sorted = np.sort(np.abs(f))
    diffs = np.diff(f_sorted)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        duration_s = float(min_duration_s)
    else:
        duration_s = float(1.0 / float(np.min(diffs)))
        duration_s = float(np.clip(duration_s, min_duration_s, max_duration_s))

    f_max = float(np.max(f_sorted))
    fs_eval = float(samples_per_period) * f_max if f_max > 0 else float(1.0 / duration_s)
    n = int(np.ceil(duration_s * fs_eval))
    n = max(n, 16)
    return np.arange(n, dtype=float) / fs_eval


def _optimise_phases_for_crest(
    *,
    freqs_hz: np.ndarray,
    amps: np.ndarray,
    phases_init_rad: Optional[np.ndarray] = None,
    fixed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Phase-only crest-factor optimisation with internal default grid selection."""
    t_s = _default_crest_time_grid_s(freqs_hz)
    return np.asarray(
        minimise_crest_factor_phases(
            freqs_hz,
            amps,
            t_s=t_s,
            passes=1,
            method="coordinate",
            output="rad",
            phases_init_rad=phases_init_rad,
            fixed_mask=fixed_mask,
        ),
        dtype=float,
    )


def _plan_segment_start_phases(
    seg: ResolvedSegment,
    *,
    logical_channel: str,
    phase_in: Optional[_PhaseContinueState],
    amp_calib: Optional[OpticalPowerToRFAmpCalib],
) -> np.ndarray:
    """Return start phases (CPU/NumPy array) for one segment/logical channel according to `seg.phase_mode`."""
    if not seg.parts:
        return np.zeros((0,), dtype=float)

    first = seg.parts[0].logical_channels[logical_channel]
    n_tones = int(first.start.freqs_hz.shape[0])

    phase_mode = str(getattr(seg, "phase_mode", "continue"))
    if phase_mode not in ("manual", "continue", "optimise"):
        raise ValueError(f"Unknown phase_mode {phase_mode!r}")

    start_phases = np.asarray(first.start.phases_rad, dtype=float).reshape(-1).copy()
    if n_tones == 0:
        return start_phases

    if phase_mode == "manual":
        return start_phases
    if phase_mode == "optimise":
        amps = np.asarray(first.start.amps, dtype=float)
        if amp_calib is not None:
            try:
                amps = np.asarray(
                    amp_calib.rf_amps(
                        np.asarray(first.start.freqs_hz, dtype=float),
                        amps,
                        logical_channel=logical_channel,
                        xp=np,
                    ),
                    dtype=float,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Optical-power calibration failed while planning segment start phases."
                ) from exc
        return _optimise_phases_for_crest(
            freqs_hz=np.asarray(first.start.freqs_hz, dtype=float),
            amps=amps,
        )

    # phase_mode == "continue"
    if phase_in is None:
        return start_phases

    mapping = _match_tones_by_frequency(
        np.asarray(phase_in.freqs_hz, dtype=float),
        np.asarray(first.start.freqs_hz, dtype=float),
    )
    fixed = np.zeros((n_tones,), dtype=bool)
    for cur_i, prev_i in mapping.items():
        start_phases[cur_i] = float(phase_in.phases_rad[prev_i])
        fixed[cur_i] = True
    if np.any(~fixed):
        amps = np.asarray(first.start.amps, dtype=float)
        if amp_calib is not None:
            try:
                amps = np.asarray(
                    amp_calib.rf_amps(
                        np.asarray(first.start.freqs_hz, dtype=float),
                        amps,
                        logical_channel=logical_channel,
                        xp=np,
                    ),
                    dtype=float,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Optical-power calibration failed while planning continued segment phases."
                ) from exc
        return _optimise_phases_for_crest(
            freqs_hz=np.asarray(first.start.freqs_hz, dtype=float),
            amps=amps,
            phases_init_rad=start_phases,
            fixed_mask=fixed,
        )
    return start_phases


def interp_logical_channel_part(
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
    logical_channel: str,
    amp_calib: Optional[OpticalPowerToRFAmpCalib],
    xp: Any = np,
) -> tuple[Any, Any]:
    """
    Synthesize one logical-channel part.

    Returns (y, phase_end) where:
    - y is (n_samples,) float waveform
    - phase_end is (n_tones,) phase at the end of the part (for continue)
    """
    freqs, amps = interp_logical_channel_part(
        pp, n_samples=n_samples, sample_rate_hz=sample_rate_hz, xp=xp
    )
    if int(freqs.shape[1]) != int(phase0_rad.shape[0]):
        raise ValueError("phase0 length mismatch with tone count")

    if freqs.shape[1] == 0:
        return xp.zeros((n_samples,), dtype=float), phase0_rad.copy()

    if amp_calib is not None:
        try:
            amps = amp_calib.rf_amps(
                freqs,
                amps,
                logical_channel=logical_channel,
                xp=xp,
            )
        except Exception as exc:
            raise RuntimeError(
                "Optical-power calibration failed during sample synthesis. "
                "If using gpu=True, ensure the calibration supports xp=cupy or compile with gpu=False."
            ) from exc

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
    phase_in: Optional[_PhaseContinueState],
    amp_calib: Optional[OpticalPowerToRFAmpCalib],
    xp: Any = np,
) -> tuple[Any, Any]:
    """Synthesize one logical channel's waveform for a full segment."""
    if not seg.parts:
        return xp.zeros((0,), dtype=float), xp.zeros((0,), dtype=float)

    phase0_np = _plan_segment_start_phases(
        seg,
        logical_channel=logical_channel,
        phase_in=phase_in,
        amp_calib=amp_calib,
    )
    phase_dtype = xp.float32 if xp is not np else float
    phase = xp.asarray(phase0_np, dtype=phase_dtype).copy()

    total = int(seg.n_samples)
    cursor = 0

    # Allocate output once to avoid list + concatenate (extra copy and peak memory).
    part0 = seg.parts[0]
    pp0 = part0.logical_channels[logical_channel]
    y0, phase = _synth_part(
        pp0,
        n_samples=part0.n_samples,
        sample_rate_hz=sample_rate_hz,
        phase0_rad=phase,
        logical_channel=logical_channel,
        amp_calib=amp_calib,
        xp=xp,
    )
    out = xp.empty((total,), dtype=y0.dtype)
    n0 = int(part0.n_samples)
    out[cursor : cursor + n0] = y0
    cursor += n0

    for part in seg.parts[1:]:
        pp = part.logical_channels[logical_channel]
        y, phase = _synth_part(
            pp,
            n_samples=part.n_samples,
            sample_rate_hz=sample_rate_hz,
            phase0_rad=phase,
            logical_channel=logical_channel,
            amp_calib=amp_calib,
            xp=xp,
        )
        n = int(part.n_samples)
        out[cursor : cursor + n] = y
        cursor += n

    return out, phase


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


def synthesized_sequence_program_to_numpy(
    prog: SynthesizedSequenceProgram,
) -> SynthesizedSequenceProgram:
    """Convert synthesized float segment buffers to CPU/NumPy arrays."""
    out_segments: list[SynthesizedSegment] = []
    for seg in prog.segments:
        data = seg.data
        if isinstance(data, np.ndarray):
            out_segments.append(seg)
            continue
        try:
            import cupy as cp  # type: ignore

            data_np = cp.asnumpy(data)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Cannot convert synthesized segment data to NumPy (CuPy not available?)."
            ) from exc
        out_segments.append(
            SynthesizedSegment(name=seg.name, n_samples=seg.n_samples, data=data_np)
        )

    if all(a is b for a, b in zip(out_segments, prog.segments, strict=True)):
        return prog

    return SynthesizedSequenceProgram(
        sample_rate_hz=prog.sample_rate_hz,
        physical_setup=prog.physical_setup,
        segments=tuple(out_segments),
        steps=tuple(prog.steps),
        quantization=tuple(prog.quantization),
    )


def compiled_sequence_program_to_numpy(
    prog: CompiledSequenceProgram,
) -> CompiledSequenceProgram:
    """Convert compiled int16 segment buffers to CPU/NumPy arrays."""
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
        physical_setup=prog.physical_setup,
        gain=float(prog.gain),
        clip=float(prog.clip),
        full_scale=int(prog.full_scale),
        segments=tuple(out_segments),
        steps=tuple(prog.steps),
        quantization=tuple(prog.quantization),
    )


def synthesize_sequence_program(
    quantized: QuantizedIR,
    *,
    physical_setup: AWGPhysicalSetupInfo,
    gpu: bool = False,
    output: Literal["numpy", "cupy"] = "numpy",
) -> SynthesizedSequenceProgram:
    """
    Synthesize float per-segment, per-channel waveform buffers.

    This stage applies optional optical-power calibration from `physical_setup`.
    The synthesized float buffers represent RF amplitudes in mV.
    It does not apply gain/clip/full_scale int16 quantization.
    """
    q_ir = quantized.resolved_ir
    q_info = quantized.quantization
    amp_calib: Optional[OpticalPowerToRFAmpCalib] = physical_setup

    if output not in ("numpy", "cupy"):
        raise ValueError("output must be 'numpy' or 'cupy'")
    if output == "cupy" and not gpu:
        raise ValueError("output='cupy' requires gpu=True")

    xp: Any = np
    cp: Any = None
    if gpu:
        cp = _cupy_or_raise()
        xp = cp

    # Validate mapping for all used logical channels before synthesis.
    for logical_channel in q_ir.logical_channels:
        physical_setup.hardware_channel(logical_channel)

    n_channels = int(physical_setup.N_ch)
    n_segments = len(q_ir.segments)
    if n_segments == 0:
        raise ValueError("No segments to synthesize")

    out_segments: list[SynthesizedSegment] = []
    out_steps: list[SequenceStep] = []

    # Phase state carried across segments, per logical channel.
    phase_state: dict[str, _PhaseContinueState] = {}

    for i, seg in enumerate(q_ir.segments):
        n = int(seg.n_samples)
        dtype = xp.float32 if xp is not np else float
        data = xp.zeros((n_channels, n), dtype=dtype)

        for logical_channel in q_ir.logical_channels:
            hw_ch = int(physical_setup.hardware_channel(logical_channel))
            y, phase_out = _synth_logical_channel_segment(
                seg,
                logical_channel=logical_channel,
                sample_rate_hz=q_ir.sample_rate_hz,
                phase_in=phase_state.get(logical_channel),
                amp_calib=amp_calib,
                xp=xp,
            )
            # Keep continue state on CPU; it is tiny compared to waveform buffers.
            if cp is not None:
                phase_out_np = cp.asnumpy(phase_out)
            else:
                phase_out_np = np.asarray(phase_out, dtype=float)
            end_freqs_np = np.asarray(
                seg.parts[-1].logical_channels[logical_channel].end.freqs_hz, dtype=float
            )
            phase_state[logical_channel] = _PhaseContinueState(
                freqs_hz=end_freqs_np, phases_rad=phase_out_np
            )
            data[hw_ch, :] = y

        if cp is not None and output == "numpy":
            data_out = cp.asnumpy(data)
        else:
            data_out = data

        out_segments.append(
            SynthesizedSegment(name=seg.name, n_samples=n, data=data_out)
        )

        on_trig = seg.mode == "wait_trig"
        loops = int(seg.loop) if seg.mode == "loop_n" else 1
        out_steps.append(
            SequenceStep(
                step_index=i,
                segment_index=i,
                next_step=(i + 1) % n_segments,
                loops=loops,
                on_trig=on_trig,
            )
        )

    return SynthesizedSequenceProgram(
        sample_rate_hz=q_ir.sample_rate_hz,
        physical_setup=physical_setup,
        segments=tuple(out_segments),
        steps=tuple(out_steps),
        quantization=tuple(q_info),
    )


def quantize_synthesized_program(
    synthesized: SynthesizedSequenceProgram,
    *,
    gain: float,
    clip: float,
    full_scale: int,
) -> CompiledSequenceProgram:
    """
    Apply gain/clip/full_scale and convert float segment buffers to int16.

    Unit convention:
    - synthesized float buffers are interpreted as RF amplitudes in mV.
    - `gain` is the normalization scale from mV to full-scale units.
      For a card configured with max output `card_max_mV`, use
      `gain = 1.0 / card_max_mV`.
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

    out_segments: list[CompiledSegment] = []
    for seg in synthesized.segments:
        data = seg.data
        if isinstance(data, np.ndarray):
            y = np.clip(float(gain) * data, -float(clip), float(clip))
            data_i16 = np.rint(y * float(full_scale)).astype(np.int16)
        else:
            try:
                import cupy as cp  # type: ignore

                y = cp.clip(float(gain) * data, -float(clip), float(clip))
                data_i16 = cp.rint(y * float(full_scale)).astype(cp.int16)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to quantize synthesized CuPy segment buffer."
                ) from exc

        out_segments.append(
            CompiledSegment(
                name=seg.name,
                n_samples=seg.n_samples,
                data_i16=data_i16,
            )
        )

    return CompiledSequenceProgram(
        sample_rate_hz=synthesized.sample_rate_hz,
        physical_setup=synthesized.physical_setup,
        gain=float(gain),
        clip=float(clip),
        full_scale=int(full_scale),
        segments=tuple(out_segments),
        steps=tuple(synthesized.steps),
        quantization=tuple(synthesized.quantization),
    )


def compile_sequence_program(
    quantized: QuantizedIR,
    *,
    physical_setup: AWGPhysicalSetupInfo,
    gain: float,
    clip: float,
    full_scale: int,
    gpu: bool = False,
    output: Literal["numpy", "cupy"] = "numpy",
) -> CompiledSequenceProgram:
    """
    Convenience wrapper: synthesize float buffers then quantize to int16.

    Performance behavior:
    - `gpu=False`: CPU synthesis + CPU quantization.
    - `gpu=True, output="cupy"`: GPU synthesis + GPU quantization, return CuPy int16.
    - `gpu=True, output="numpy"` (default): run synthesis+quantization on GPU, then
      transfer final int16 buffers to NumPy. This avoids transferring intermediate
      float buffers and is typically the fastest path for NumPy output.

    Scaling convention:
    - `gain` is the normalization factor passed to `quantize_synthesized_program`.
      If synthesized amplitudes are in mV and AWG full-scale corresponds to
      `card_max_mV`, set `gain = 1.0 / card_max_mV`.
    """
    if gpu and output == "numpy":
        synthesized_gpu = synthesize_sequence_program(
            quantized,
            physical_setup=physical_setup,
            gpu=True,
            output="cupy",
        )
        compiled_gpu = quantize_synthesized_program(
            synthesized_gpu,
            gain=gain,
            clip=clip,
            full_scale=full_scale,
        )
        return compiled_sequence_program_to_numpy(compiled_gpu)

    synthesized = synthesize_sequence_program(
        quantized,
        physical_setup=physical_setup,
        gpu=gpu,
        output=output,
    )
    return quantize_synthesized_program(
        synthesized,
        gain=gain,
        clip=clip,
        full_scale=full_scale,
    )
