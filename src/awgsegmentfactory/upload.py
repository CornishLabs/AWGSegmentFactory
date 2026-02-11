"""Upload helpers (compiled segments -> hardware).

This module intentionally keeps the public API stable while hardware backends evolve.

Today:
- "cpu" upload is typically done using Spectrum's normal DMA APIs (host buffers).

Future:
- "scapp" upload will target Spectrum SCAPP / RDMA (GPU buffers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .synth_samples import CompiledSequenceProgram, compiled_sequence_program_to_numpy


UploadMode = Literal["cpu", "scapp", "auto"]


@dataclass(frozen=True)
class CPUUploadSession:
    """
    Reusable CPU-upload session for data-only segment updates.

    Keep this object between uploads when the sequence graph (segment lengths + steps)
    is unchanged and only segment sample data changes.
    """

    card: Any
    sequence: Any
    segments_hw: tuple[Any, ...]
    steps_hw: tuple[Any, ...]
    n_channels: int
    segment_lengths: tuple[int, ...]
    steps_signature: tuple[tuple[int, int, int, int, bool], ...]


def _steps_signature(
    prog: CompiledSequenceProgram,
) -> tuple[tuple[int, int, int, int, bool], ...]:
    return tuple(
        (
            int(s.step_index),
            int(s.segment_index),
            int(s.next_step),
            int(s.loops),
            bool(s.on_trig),
        )
        for s in prog.steps
    )


def _as_numpy_i16_segment_data(
    data_i16: Any,
    *,
    n_channels: int,
    n_samples: int,
) -> np.ndarray:
    arr = np.asarray(data_i16)
    if arr.dtype != np.int16:
        raise ValueError(f"Segment data must be int16, got {arr.dtype}")
    expected = (int(n_channels), int(n_samples))
    if arr.shape != expected:
        raise ValueError(
            f"Segment data shape mismatch: expected {expected}, got {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _ensure_numpy_compiled(prog: CompiledSequenceProgram) -> CompiledSequenceProgram:
    kinds = {
        "numpy" if isinstance(seg.data_i16, np.ndarray) else "other"
        for seg in prog.segments
    }
    if len(kinds) != 1:
        raise ValueError(
            "Compiled program has mixed segment buffer types; expected all-NumPy or all-CuPy."
        )
    if "numpy" in kinds:
        return prog
    return compiled_sequence_program_to_numpy(prog)


def _full_cpu_upload(
    prog: CompiledSequenceProgram,
    *,
    card: Any,
) -> CPUUploadSession:
    import spcm

    sequence = spcm.Sequence(card)
    n_channels = int(prog.physical_setup.N_ch)

    segments_hw: list[Any] = []
    for seg in prog.segments:
        data_i16 = _as_numpy_i16_segment_data(
            seg.data_i16,
            n_channels=n_channels,
            n_samples=int(seg.n_samples),
        )
        s = sequence.add_segment(int(seg.n_samples))
        s[:, :] = data_i16
        segments_hw.append(s)

    if not prog.steps:
        raise ValueError("Cannot upload: compiled program has no steps")

    steps_hw: list[Any] = []
    for step in prog.steps:
        seg_idx = int(step.segment_index)
        if not (0 <= seg_idx < len(segments_hw)):
            raise ValueError(
                f"Step {step.step_index} references invalid segment_index={seg_idx}"
            )
        steps_hw.append(
            sequence.add_step(segments_hw[seg_idx], loops=int(step.loops))
        )

    sequence.entry_step(steps_hw[0])

    for step in prog.steps:
        i = int(step.step_index)
        j = int(step.next_step)
        if not (0 <= i < len(steps_hw)):
            raise ValueError(f"Invalid step_index={i} for step table size {len(steps_hw)}")
        if not (0 <= j < len(steps_hw)):
            raise ValueError(f"Invalid next_step={j} for step table size {len(steps_hw)}")
        steps_hw[i].set_transition(steps_hw[j], on_trig=bool(step.on_trig))

    sequence.write_setup()

    return CPUUploadSession(
        card=card,
        sequence=sequence,
        segments_hw=tuple(segments_hw),
        steps_hw=tuple(steps_hw),
        n_channels=n_channels,
        segment_lengths=tuple(int(seg.n_samples) for seg in prog.segments),
        steps_signature=_steps_signature(prog),
    )


def _update_cpu_segments_only(
    prog: CompiledSequenceProgram,
    *,
    session: CPUUploadSession,
    segment_indices: Sequence[int] | None,
) -> CPUUploadSession:
    if session.card is None:
        raise ValueError("CPUUploadSession.card is None")
    if len(session.segments_hw) != len(prog.segments):
        raise ValueError(
            "Segment count changed; data-only update is invalid. "
            "Do a full upload (cpu_session=None)."
        )

    expected_lengths = tuple(int(seg.n_samples) for seg in prog.segments)
    if session.segment_lengths != expected_lengths:
        raise ValueError(
            "Segment lengths changed; data-only update is invalid. "
            "Do a full upload (cpu_session=None)."
        )

    if session.steps_signature != _steps_signature(prog):
        raise ValueError(
            "Step transition graph changed; data-only update is invalid. "
            "Do a full upload (cpu_session=None)."
        )

    if int(session.n_channels) != int(prog.physical_setup.N_ch):
        raise ValueError(
            "Channel count changed; data-only update is invalid. "
            "Do a full upload (cpu_session=None)."
        )

    if segment_indices is None:
        indices = list(range(len(prog.segments)))
    else:
        indices = sorted(set(int(i) for i in segment_indices))
        if not indices:
            raise ValueError("segment_indices must be non-empty when provided")
    for idx in indices:
        if not (0 <= idx < len(prog.segments)):
            raise ValueError(
                f"segment_indices contains out-of-range index {idx} "
                f"for {len(prog.segments)} segments"
            )

    for idx in indices:
        seg = prog.segments[idx]
        data_i16 = _as_numpy_i16_segment_data(
            seg.data_i16,
            n_channels=session.n_channels,
            n_samples=int(seg.n_samples),
        )
        seg_hw = session.segments_hw[idx]
        seg_hw[:, :] = data_i16
        session.sequence.transfer_segment(seg_hw)

    return session


def upload_sequence_program(
    prog: CompiledSequenceProgram,
    *,
    mode: UploadMode = "cpu",
    card: Any = None,
    cpu_session: CPUUploadSession | None = None,
    segment_indices: Sequence[int] | None = None,
) -> CPUUploadSession:
    """
    Upload a compiled sequence program to hardware.

    Parameters
    ----------
    prog:
        Output of `compile_sequence_program(...)`. Segment buffers may be NumPy or CuPy
        depending on `output=...`.
    mode:
        - "cpu": host-buffer upload (NumPy). If buffers are CuPy, convert to NumPy first.
        - "scapp": GPU-buffer upload via SCAPP/RDMA (requires CuPy buffers).
        - "auto": select "scapp" if buffers are CuPy, else "cpu".
    card:
        Open hardware card handle (required for "cpu" uploads).
    cpu_session:
        Existing CPU upload session returned by a prior full upload. When provided,
        upload runs in data-only mode (segment content rewrite) and keeps step graph.
    segment_indices:
        Optional list of segment indices to update in data-only mode. If omitted,
        all segment data are rewritten.

    Returns
    -------
    CPUUploadSession
        Session handle to reuse for later data-only segment updates.
    """
    if mode not in ("cpu", "scapp", "auto"):
        raise ValueError("mode must be 'cpu', 'scapp', or 'auto'")

    if not prog.segments:
        raise ValueError("Cannot upload: compiled program has no segments")

    kinds = {
        "numpy" if isinstance(seg.data_i16, np.ndarray) else "other"
        for seg in prog.segments
    }
    if len(kinds) != 1:
        raise ValueError(
            "Compiled program has mixed segment buffer types; expected all-NumPy or all-CuPy."
        )
    is_numpy = ("numpy" in kinds) and (len(kinds) == 1)

    if mode == "auto":
        mode = "cpu" if is_numpy else "scapp"

    if mode == "cpu":
        if card is None:
            raise ValueError("CPU upload requires an open `card` handle")
        prog_cpu = _ensure_numpy_compiled(prog)
        if cpu_session is None:
            return _full_cpu_upload(prog_cpu, card=card)
        if cpu_session.card is not card:
            raise ValueError(
                "cpu_session belongs to a different card object; "
                "create a new session with a full upload."
            )
        return _update_cpu_segments_only(
            prog_cpu,
            session=cpu_session,
            segment_indices=segment_indices,
        )

    # mode == "scapp"
    if is_numpy:
        raise ValueError(
            "SCAPP upload requires GPU-resident CuPy buffers. "
            "Compile with gpu=True and output='cupy'."
        )
    raise NotImplementedError(
        "SCAPP/RDMA upload for sequence-mode segments is not implemented yet."
    )
