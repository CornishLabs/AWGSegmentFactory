"""Upload helpers (compiled segment slots -> hardware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .synth_samples import QIRtoSamplesSegmentCompiler, compiled_sequence_slots_to_numpy


UploadMode = Literal["cpu", "scapp", "auto"]


@dataclass(frozen=True)
class CPUUploadSession:
    """
    Reusable CPU-upload session for segment data updates.

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
    repo: QIRtoSamplesSegmentCompiler,
) -> tuple[tuple[int, int, int, int, bool], ...]:
    return tuple(
        (
            int(s.step_index),
            int(s.segment_index),
            int(s.next_step),
            int(s.loops),
            bool(s.on_trig),
        )
        for s in repo.steps
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


def _segment_lengths_from_repo(repo: QIRtoSamplesSegmentCompiler) -> tuple[int, ...]:
    return tuple(int(seg.n_samples) for seg in repo.quantized.segments)


def _ensure_numpy_repo(repo: QIRtoSamplesSegmentCompiler) -> QIRtoSamplesSegmentCompiler:
    return compiled_sequence_slots_to_numpy(repo)


def _full_cpu_upload(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    card: Any,
) -> CPUUploadSession:
    import spcm

    sequence = spcm.Sequence(card)
    n_channels = int(repo.physical_setup.N_ch)
    segment_lengths = _segment_lengths_from_repo(repo)

    compiled_segments = repo.segments
    if len(compiled_segments) != len(segment_lengths):
        raise ValueError(
            "Full upload requires all segments to be compiled. "
            "Compile all segments first."
        )

    segments_hw: list[Any] = []
    for idx, seg in enumerate(compiled_segments):
        data_i16 = _as_numpy_i16_segment_data(
            seg.data_i16,
            n_channels=n_channels,
            n_samples=segment_lengths[idx],
        )
        s = sequence.add_segment(segment_lengths[idx])
        s[:, :] = data_i16
        segments_hw.append(s)

    if not repo.steps:
        raise ValueError("Cannot upload: compiled program has no steps")

    steps_hw: list[Any] = []
    for step in repo.steps:
        seg_idx = int(step.segment_index)
        if not (0 <= seg_idx < len(segments_hw)):
            raise ValueError(
                f"Step {step.step_index} references invalid segment_index={seg_idx}"
            )
        steps_hw.append(sequence.add_step(segments_hw[seg_idx], loops=int(step.loops)))

    sequence.entry_step(steps_hw[0])

    for step in repo.steps:
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
        segment_lengths=segment_lengths,
        steps_signature=_steps_signature(repo),
    )


def _update_cpu_segments_only(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    session: CPUUploadSession,
    segment_indices: Sequence[int] | None,
) -> CPUUploadSession:
    if session.card is None:
        raise ValueError("CPUUploadSession.card is None")

    expected_lengths = _segment_lengths_from_repo(repo)
    if session.segment_lengths != expected_lengths:
        raise ValueError(
            "Segment lengths changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )
    if session.steps_signature != _steps_signature(repo):
        raise ValueError(
            "Step transition graph changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )
    if int(session.n_channels) != int(repo.physical_setup.N_ch):
        raise ValueError(
            "Channel count changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )

    if segment_indices is None:
        indices = list(repo.compiled_indices)
        if not indices:
            raise ValueError(
                "No compiled segments available to update. "
                "Compile at least one segment first."
            )
    else:
        indices = sorted(set(int(i) for i in segment_indices))
        if not indices:
            raise ValueError("segment_indices must be non-empty when provided")

    for idx in indices:
        if not (0 <= idx < len(session.segment_lengths)):
            raise ValueError(
                f"segment_indices contains out-of-range index {idx} "
                f"for {len(session.segment_lengths)} segments"
            )

    for idx in indices:
        seg = repo.compiled_segment(idx)
        data_i16 = _as_numpy_i16_segment_data(
            seg.data_i16,
            n_channels=session.n_channels,
            n_samples=session.segment_lengths[idx],
        )
        seg_hw = session.segments_hw[idx]
        seg_hw[:, :] = data_i16
        session.sequence.transfer_segment(seg_hw)

    return session


def upload_sequence_program(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    mode: UploadMode = "cpu",
    card: Any = None,
    cpu_session: CPUUploadSession | None = None,
    segment_indices: Sequence[int] | None = None,
    upload_steps: bool = True,
) -> CPUUploadSession:
    """
    Upload compiled sequence slots to hardware.

    Parameters
    ----------
    repo:
        Compiled slot container (`QIRtoSamplesSegmentCompiler`).
    mode:
        - "cpu": host-buffer upload (NumPy). If buffers are CuPy, convert to NumPy.
        - "scapp": GPU-buffer upload via SCAPP/RDMA (future).
        - "auto": select "scapp" if compiled buffers are GPU-resident, else "cpu".
    card:
        Open hardware card handle. Required for full CPU upload (`upload_steps=True`).
    cpu_session:
        Existing session from a prior full upload. Required for data-only updates
        (`upload_steps=False`).
    segment_indices:
        Optional segment indices to update for data-only mode. If omitted in data-only
        mode, all currently compiled segments in `repo` are uploaded.
    upload_steps:
        - True (default): full upload of all segments + step graph (`sequence.write_setup()`).
          Requires all segments compiled.
        - False: data-only segment update using existing `cpu_session`.

    Returns
    -------
    CPUUploadSession
        Session handle to reuse for later data-only updates.
    """
    if mode not in ("cpu", "scapp", "auto"):
        raise ValueError("mode must be 'cpu', 'scapp', or 'auto'")

    compiled_items = repo.compiled_segment_items()
    if not compiled_items:
        raise ValueError("Cannot upload: no compiled segment slots available")

    is_numpy = all(isinstance(seg.data_i16, np.ndarray) for _i, seg in compiled_items)
    if mode == "auto":
        mode = "cpu" if is_numpy else "scapp"

    if mode == "cpu":
        repo_cpu = _ensure_numpy_repo(repo)
        if upload_steps:
            if segment_indices is not None:
                raise ValueError(
                    "segment_indices is only valid when upload_steps=False"
                )
            if card is None:
                raise ValueError("Full CPU upload requires an open `card` handle")
            return _full_cpu_upload(repo_cpu, card=card)

        if cpu_session is None:
            raise ValueError(
                "Data-only upload requires `cpu_session` from a prior full upload"
            )
        if card is not None and cpu_session.card is not card:
            raise ValueError(
                "cpu_session belongs to a different card object than `card`"
            )
        return _update_cpu_segments_only(
            repo_cpu,
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
