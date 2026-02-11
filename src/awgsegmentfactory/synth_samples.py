"""Sample synthesis and quantization for AWG sequence slots.

Primary API:
- `QIRtoSamplesSegmentCompiler.initialise_from_quantised(...)` creates a slot container.
- `QIRtoSamplesSegmentCompiler.compile(...)` compiles all or selected segments.
- `compile_sequence_program(...)` is a convenience wrapper for full compile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np

from .calibration import AWGPhysicalSetupInfo, OpticalPowerToRFAmpCalib
from .interpolation import interp_param
from .phase_minimiser import minimise_crest_factor_phases
from .quantize import QuantizedIR, SegmentQuantizationInfo
from .resolved_ir import ResolvedLogicalChannelPart, ResolvedSegment


@dataclass(frozen=True)
class CompiledSegment:
    """One quantized hardware segment: int16 array per hardware channel."""

    name: str
    n_samples: int
    # Shape: (n_channels, n_samples), NumPy or CuPy depending on compile output.
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
class _PhaseContinueState:
    """Per-logical-channel state needed to continue phases across segments."""

    freqs_hz: np.ndarray
    phases_rad: np.ndarray


def _match_tones_by_frequency(
    prev_freqs_hz: np.ndarray,
    cur_freqs_hz: np.ndarray,
    *,
    tol_hz: float = 1.0,
) -> dict[int, int]:
    """Return mapping `{cur_idx -> prev_idx}` by greedy nearest-frequency matching."""
    prev = np.asarray(prev_freqs_hz, dtype=float).reshape(-1)
    cur = np.asarray(cur_freqs_hz, dtype=float).reshape(-1)
    if prev.size == 0 or cur.size == 0:
        return {}

    d = np.abs(prev[:, None] - cur[None, :])
    pairs = np.argwhere(d <= float(tol_hz))
    if pairs.size == 0:
        return {}

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
    """Return start phases for one segment/logical channel according to `seg.phase_mode`."""
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
    """Return `(freqs_hz, amps)` as `(n_samples, n_tones)` arrays."""
    f0 = pp.start.freqs_hz
    f1 = pp.end.freqs_hz
    a0 = pp.start.amps
    a1 = pp.end.amps

    if f0.shape != f1.shape or a0.shape != a1.shape:
        raise ValueError("Start/end shape mismatch for logical channel part")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    n_tones = int(f0.shape[0])
    if n_tones == 0:
        z = xp.zeros((n_samples, 0), dtype=float)
        return z, z

    if pp.interp.kind == "hold":
        dtype = xp.float32 if xp is not np else float
        freqs = xp.repeat(xp.asarray(f0, dtype=dtype)[None, :], n_samples, axis=0)
        amps = xp.repeat(xp.asarray(a0, dtype=dtype)[None, :], n_samples, axis=0)
        return freqs, amps

    dtype = xp.float32 if xp is not np else float
    u = (xp.arange(n_samples, dtype=dtype) / float(n_samples))[:, None]
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
    """Synthesize one logical-channel part."""
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
    dphi *= float(2.0 * np.pi * dt)
    phase_end = (phase0_rad + xp.sum(dphi, axis=0)) % (2.0 * np.pi)

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


def _quantise_voltage_buffer(
    data: Any,
    *,
    full_scale_mv: float,
    full_scale: int,
    clip: float,
) -> Any:
    """Convert synthesized RF-voltage buffers (mV) into int16 AWG samples."""
    if full_scale_mv <= 0:
        raise ValueError("full_scale_mv must be > 0")
    if not (0 < clip <= 1.0):
        raise ValueError("clip must be in (0, 1]")
    if full_scale <= 0:
        raise ValueError("full_scale must be > 0")
    max_i16 = int(np.iinfo(np.int16).max)
    if full_scale > max_i16:
        raise ValueError(f"full_scale must be <= {max_i16} for int16 output")

    scale = 1.0 / float(full_scale_mv)
    if isinstance(data, np.ndarray):
        y = np.clip(scale * data, -float(clip), float(clip))
        return np.rint(y * float(full_scale)).astype(np.int16)

    try:
        import cupy as cp  # type: ignore

        y = cp.clip(scale * data, -float(clip), float(clip))
        return cp.rint(y * float(full_scale)).astype(cp.int16)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to quantize synthesized CuPy segment buffer.") from exc


def _build_sequence_steps(segments: Sequence[ResolvedSegment]) -> tuple[SequenceStep, ...]:
    n_segments = len(segments)
    out: list[SequenceStep] = []
    for i, seg in enumerate(segments):
        on_trig = seg.mode == "wait_trig"
        loops = int(seg.loop) if seg.mode == "loop_n" else 1
        out.append(
            SequenceStep(
                step_index=i,
                segment_index=i,
                next_step=(i + 1) % n_segments,
                loops=loops,
                on_trig=on_trig,
            )
        )
    return tuple(out)


class QIRtoSamplesSegmentCompiler:
    """
    Slot container for compiled segment data.

    Compile all segments:
        `repo.compile()`

    Recompile only selected segments (contiguous run):
        `repo.compile(segment_indices=[k])`
        `repo.compile(segment_indices=[k, k + 1])`
        `repo.compile(segment_names=["seg_name"])`

    Notes:
    - Segment selection is intentionally restricted to a contiguous index run to keep
      phase continuity handling explicit and deterministic.
    - If selected segments use `phase_mode="continue"` and require predecessor phase
      state, compile either:
        - with predecessor segments already compiled in this repo, or
        - with `phase_seed=<another QIRtoSamplesSegmentCompiler>` carrying predecessor state.
    """

    def __init__(
        self,
        *,
        quantized: QuantizedIR,
        physical_setup: AWGPhysicalSetupInfo,
        full_scale_mv: float,
        full_scale: int,
        clip: float = 1.0,
    ) -> None:
        self.quantized = quantized
        self.physical_setup = physical_setup
        self.full_scale_mv = float(full_scale_mv)
        self.full_scale = int(full_scale)
        self.clip = float(clip)

        if self.full_scale_mv <= 0:
            raise ValueError("full_scale_mv must be > 0")
        if self.full_scale <= 0:
            raise ValueError("full_scale must be > 0")
        if not (0 < self.clip <= 1.0):
            raise ValueError("clip must be in (0, 1]")

        q_ir = self.quantized.resolved_ir
        if not q_ir.segments:
            raise ValueError("No segments to compile")
        for logical_channel in q_ir.logical_channels:
            self.physical_setup.hardware_channel(logical_channel)

        self._steps: tuple[SequenceStep, ...] = _build_sequence_steps(q_ir.segments)
        self._segments: list[CompiledSegment | None] = [None] * len(q_ir.segments)
        self._phase_end_states: list[dict[str, _PhaseContinueState] | None] = [
            None
        ] * len(q_ir.segments)

    @classmethod
    def initialise_from_quantised(
        cls,
        *,
        quantized: QuantizedIR,
        physical_setup: AWGPhysicalSetupInfo,
        full_scale_mv: float,
        full_scale: int,
        clip: float = 1.0,
    ) -> "QIRtoSamplesSegmentCompiler":
        return cls(
            quantized=quantized,
            physical_setup=physical_setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
            clip=clip,
        )

    @property
    def sample_rate_hz(self) -> float:
        return float(self.quantized.sample_rate_hz)

    @property
    def quantization(self) -> tuple[SegmentQuantizationInfo, ...]:
        return tuple(self.quantized.quantization)

    @property
    def steps(self) -> tuple[SequenceStep, ...]:
        return self._steps

    @property
    def n_segments(self) -> int:
        return len(self.quantized.segments)

    @property
    def segment_names(self) -> tuple[str, ...]:
        return tuple(str(seg.name) for seg in self.quantized.segments)

    @property
    def compiled_indices(self) -> tuple[int, ...]:
        return tuple(i for i, seg in enumerate(self._segments) if seg is not None)

    @property
    def segments(self) -> tuple[CompiledSegment, ...]:
        missing = [i for i, seg in enumerate(self._segments) if seg is None]
        if missing:
            raise ValueError(
                "Not all segments are compiled. Missing indices: "
                f"{missing}. Compile them first or use `compiled_segment_items(...)`."
            )
        return tuple(seg for seg in self._segments if seg is not None)

    def segment_index(self, name: str) -> int:
        key = str(name)
        names = self.segment_names
        for i, nm in enumerate(names):
            if nm == key:
                return i
        raise KeyError(f"Unknown segment name {key!r}; available: {list(names)}")

    def compiled_segment(self, idx: int) -> CompiledSegment:
        i = int(idx)
        if not (0 <= i < self.n_segments):
            raise IndexError(f"Segment index out of range: {i}")
        seg = self._segments[i]
        if seg is None:
            raise ValueError(f"Segment {i} is not compiled")
        return seg

    def compiled_segment_items(
        self,
        segment_indices: Sequence[int] | None = None,
    ) -> tuple[tuple[int, CompiledSegment], ...]:
        if segment_indices is None:
            return tuple((i, seg) for i, seg in enumerate(self._segments) if seg is not None)

        out: list[tuple[int, CompiledSegment]] = []
        indices = sorted(set(int(i) for i in segment_indices))
        if not indices:
            raise ValueError("segment_indices must be non-empty when provided")
        for idx in indices:
            out.append((idx, self.compiled_segment(idx)))
        return tuple(out)

    def _buffer_kind(self) -> Literal["empty", "numpy", "cupy", "mixed"]:
        kinds: set[str] = set()
        for seg in self._segments:
            if seg is None:
                continue
            kinds.add("numpy" if isinstance(seg.data_i16, np.ndarray) else "other")
        if not kinds:
            return "empty"
        if len(kinds) > 1:
            return "mixed"
        return "numpy" if "numpy" in kinds else "cupy"

    def _resolve_segment_indices(
        self,
        *,
        segment_indices: Sequence[int] | None,
        segment_names: Sequence[str] | None,
    ) -> list[int]:
        if segment_indices is not None and segment_names is not None:
            raise ValueError("Provide either segment_indices or segment_names, not both")

        if segment_indices is None and segment_names is None:
            return list(range(self.n_segments))

        if segment_names is not None:
            idxs = sorted(set(self.segment_index(name) for name in segment_names))
        else:
            idxs = sorted(set(int(i) for i in segment_indices or ()))

        if not idxs:
            raise ValueError("No segments selected for compilation")
        for idx in idxs:
            if not (0 <= idx < self.n_segments):
                raise ValueError(
                    f"segment index {idx} is out of range for {self.n_segments} segments"
                )

        expected = list(range(idxs[0], idxs[-1] + 1))
        if idxs != expected:
            raise ValueError(
                "Selected segments must form one contiguous run. "
                f"Got {idxs}, expected contiguous {expected}"
            )
        return idxs

    def _phase_seed_for_channel(
        self,
        *,
        segment_index: int,
        logical_channel: str,
        phase_seed: "QIRtoSamplesSegmentCompiler | None",
    ) -> _PhaseContinueState | None:
        if segment_index <= 0:
            return None

        prev = self._phase_end_states[segment_index - 1]
        if prev is not None:
            return prev.get(logical_channel)

        if phase_seed is None:
            return None
        if phase_seed.n_segments != self.n_segments:
            raise ValueError(
                "phase_seed has a different number of segments; cannot reuse phase state"
            )
        prev_seed = phase_seed._phase_end_states[segment_index - 1]
        if prev_seed is None:
            return None
        return prev_seed.get(logical_channel)

    def compile(
        self,
        *,
        segment_indices: Sequence[int] | None = None,
        segment_names: Sequence[str] | None = None,
        phase_seed: "QIRtoSamplesSegmentCompiler | None" = None,
        require_phase_seed_for_continue: bool = True,
        gpu: bool = False,
        output: Literal["numpy", "cupy"] = "numpy",
    ) -> "QIRtoSamplesSegmentCompiler":
        """
        Compile selected segments into slot buffers.

        Parameters
        ----------
        segment_indices / segment_names:
            Choose which segments to compile. If omitted, compiles all segments.
            Selection must be one contiguous run.
        phase_seed:
            Optional previously compiled slots object used only as phase-state seed
            for `phase_mode="continue"` predecessor continuity.
        require_phase_seed_for_continue:
            If True (default), compiling a non-initial `continue` segment without a
            known predecessor phase state raises.
        gpu / output:
            Same semantics as the old compile API:
            - `gpu=False`: CPU synthesis + quantization.
            - `gpu=True, output="cupy"`: GPU synthesis + quantization (CuPy buffers).
            - `gpu=True, output="numpy"`: GPU synthesis + quantization + final copy to NumPy.
        """
        if output not in ("numpy", "cupy"):
            raise ValueError("output must be 'numpy' or 'cupy'")
        if output == "cupy" and not gpu:
            raise ValueError("output='cupy' requires gpu=True")

        idxs = self._resolve_segment_indices(
            segment_indices=segment_indices,
            segment_names=segment_names,
        )

        target_kind = "cupy" if output == "cupy" else "numpy"
        existing_kind = self._buffer_kind()
        if existing_kind == "mixed":
            raise RuntimeError(
                "Compiled slots contain mixed buffer types; create a fresh slots object."
            )
        if existing_kind in ("numpy", "cupy") and existing_kind != target_kind:
            raise ValueError(
                "Cannot mix compiled buffer types in one slots object. "
                f"Existing={existing_kind}, requested={target_kind}. "
                "Create a fresh QIRtoSamplesSegmentCompiler for the other output type."
            )

        xp: Any = np
        cp: Any = None
        if gpu:
            cp = _cupy_or_raise()
            xp = cp

        q_ir = self.quantized.resolved_ir
        amp_calib: Optional[OpticalPowerToRFAmpCalib] = self.physical_setup
        n_channels = int(self.physical_setup.N_ch)

        for idx in idxs:
            seg = q_ir.segments[idx]
            n = int(seg.n_samples)
            dtype = xp.float32 if xp is not np else float
            data = xp.zeros((n_channels, n), dtype=dtype)

            phase_out_by_channel: dict[str, _PhaseContinueState] = {}
            phase_mode = str(getattr(seg, "phase_mode", "continue"))
            for logical_channel in q_ir.logical_channels:
                phase_in: _PhaseContinueState | None = None
                if idx > 0 and phase_mode == "continue":
                    phase_in = self._phase_seed_for_channel(
                        segment_index=idx,
                        logical_channel=logical_channel,
                        phase_seed=phase_seed,
                    )
                    if phase_in is None and require_phase_seed_for_continue:
                        raise ValueError(
                            "Cannot compile segment without predecessor phase state: "
                            f"index={idx} name={seg.name!r} logical_channel={logical_channel!r}. "
                            "Compile a predecessor prefix first or pass phase_seed with "
                            f"segment {idx - 1} already compiled."
                        )

                hw_ch = int(self.physical_setup.hardware_channel(logical_channel))
                y, phase_out = _synth_logical_channel_segment(
                    seg,
                    logical_channel=logical_channel,
                    sample_rate_hz=q_ir.sample_rate_hz,
                    phase_in=phase_in,
                    amp_calib=amp_calib,
                    xp=xp,
                )
                data[hw_ch, :] = y

                if cp is not None:
                    phase_out_np = cp.asnumpy(phase_out)
                else:
                    phase_out_np = np.asarray(phase_out, dtype=float)
                end_freqs_np = np.asarray(
                    seg.parts[-1].logical_channels[logical_channel].end.freqs_hz,
                    dtype=float,
                )
                phase_out_by_channel[logical_channel] = _PhaseContinueState(
                    freqs_hz=end_freqs_np,
                    phases_rad=phase_out_np,
                )

            data_i16 = _quantise_voltage_buffer(
                data,
                full_scale_mv=self.full_scale_mv,
                full_scale=self.full_scale,
                clip=self.clip,
            )
            if cp is not None and output == "numpy":
                data_i16 = cp.asnumpy(data_i16)

            self._segments[idx] = CompiledSegment(
                name=seg.name,
                n_samples=n,
                data_i16=data_i16,
            )
            self._phase_end_states[idx] = phase_out_by_channel

        # Any downstream segment that wasn't part of this compile run must be considered stale.
        last = idxs[-1]
        for j in range(last + 1, self.n_segments):
            self._segments[j] = None
            self._phase_end_states[j] = None

        return self

    def to_numpy(self) -> "QIRtoSamplesSegmentCompiler":
        """Return a copy with all compiled segment buffers on CPU/NumPy."""
        out = QIRtoSamplesSegmentCompiler.initialise_from_quantised(
            quantized=self.quantized,
            physical_setup=self.physical_setup,
            full_scale_mv=self.full_scale_mv,
            full_scale=self.full_scale,
            clip=self.clip,
        )

        for i, seg in enumerate(self._segments):
            if seg is None:
                continue
            data = seg.data_i16
            if isinstance(data, np.ndarray):
                data_np = data
            else:
                try:
                    import cupy as cp  # type: ignore

                    data_np = cp.asnumpy(data)
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Cannot convert compiled segment data to NumPy (CuPy unavailable?)."
                    ) from exc
            out._segments[i] = CompiledSegment(
                name=seg.name,
                n_samples=seg.n_samples,
                data_i16=data_np,
            )

        for i, st in enumerate(self._phase_end_states):
            if st is None:
                continue
            copied: dict[str, _PhaseContinueState] = {}
            for lc, s in st.items():
                copied[lc] = _PhaseContinueState(
                    freqs_hz=np.asarray(s.freqs_hz, dtype=float).copy(),
                    phases_rad=np.asarray(s.phases_rad, dtype=float).copy(),
                )
            out._phase_end_states[i] = copied

        return out


def compiled_sequence_slots_to_numpy(
    repo: QIRtoSamplesSegmentCompiler,
) -> QIRtoSamplesSegmentCompiler:
    """Convert any compiled CuPy slot buffers to NumPy."""
    return repo.to_numpy()


def quantise_and_normalise_voltage_for_awg(
    synthesised_mV: Any,
    *,
    full_scale_mv: float,
    full_scale: int,
    clip: float = 1.0,
) -> Any:
    """
    Quantize one synthesized voltage buffer (`mV`) to AWG int16 codes.

    This helper is now buffer-level only. For sequence compilation, use
    `QIRtoSamplesSegmentCompiler.compile(...)`.
    """
    return _quantise_voltage_buffer(
        synthesised_mV,
        full_scale_mv=full_scale_mv,
        full_scale=full_scale,
        clip=clip,
    )


def compile_sequence_program(
    quantized: QuantizedIR,
    *,
    physical_setup: AWGPhysicalSetupInfo,
    full_scale_mv: float,
    full_scale: int,
    clip: float = 1.0,
    gpu: bool = False,
    output: Literal["numpy", "cupy"] = "numpy",
) -> QIRtoSamplesSegmentCompiler:
    """Convenience wrapper: create slots and compile all segments."""
    repo = QIRtoSamplesSegmentCompiler.initialise_from_quantised(
        quantized=quantized,
        physical_setup=physical_setup,
        full_scale_mv=full_scale_mv,
        full_scale=full_scale,
        clip=clip,
    )
    return repo.compile(gpu=gpu, output=output)
