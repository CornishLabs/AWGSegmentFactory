"""Upload helpers (compiled segment slots -> hardware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .synth_samples import QIRtoSamplesSegmentCompiler


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


@dataclass(frozen=True)
class SCAPPUploadSession:
    """
    Reusable SCAPP-upload session for GPU-resident segment data updates.

    Keep this object between uploads when the sequence graph (segment lengths + steps)
    is unchanged and only segment sample data changes.
    """

    card: Any
    n_channels: int
    segment_lengths: tuple[int, ...]
    steps_signature: tuple[tuple[int, int, int, int, bool], ...]


UploadSession = CPUUploadSession | SCAPPUploadSession


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
    return tuple(int(seg.n_samples) for seg in repo.quantised.segments)


def _segment_count_pow2(n: int) -> int:
    n_int = int(n)
    if n_int <= 0:
        raise ValueError(f"Segment count must be >= 1, got {n_int}")
    p = 1
    while p < n_int:
        p <<= 1
    return p


def _ensure_numpy_repo(repo: QIRtoSamplesSegmentCompiler) -> QIRtoSamplesSegmentCompiler:
    return repo.to_numpy()


def _resolve_upload_indices(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    segment_indices: Sequence[int] | None,
) -> list[int]:
    if segment_indices is None:
        indices = list(repo.compiled_indices)
        if not indices:
            raise ValueError(
                "No compiled segments available to update. "
                "Compile at least one segment first."
            )
        return indices

    indices = sorted(set(int(i) for i in segment_indices))
    if not indices:
        raise ValueError("segment_indices must be non-empty when provided")
    return indices


def _validate_session_compat(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    n_channels: int,
    segment_lengths: tuple[int, ...],
    steps_signature: tuple[tuple[int, int, int, int, bool], ...],
) -> None:
    expected_lengths = _segment_lengths_from_repo(repo)
    if segment_lengths != expected_lengths:
        raise ValueError(
            "Segment lengths changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )
    if steps_signature != _steps_signature(repo):
        raise ValueError(
            "Step transition graph changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )
    if int(n_channels) != int(repo.physical_setup.N_ch):
        raise ValueError(
            "Channel count changed; data-only update is invalid. "
            "Do a full upload (upload_steps=True)."
        )


def _as_cupy_i16_segment_data(
    data_i16: Any,
    *,
    n_channels: int,
    n_samples: int,
) -> Any:
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SCAPP upload requires CuPy (compile with gpu=True, output='cupy')."
        ) from exc

    if not isinstance(data_i16, cp.ndarray):
        raise ValueError(
            "SCAPP upload requires GPU-resident CuPy segment buffers. "
            f"Got {type(data_i16)!r}."
        )
    if data_i16.dtype != cp.int16:
        raise ValueError(f"Segment data must be int16, got {data_i16.dtype}")
    expected = (int(n_channels), int(n_samples))
    if tuple(data_i16.shape) != expected:
        raise ValueError(
            f"Segment data shape mismatch: expected {expected}, got {tuple(data_i16.shape)}"
        )
    return cp.asfortranarray(data_i16)


def _upload_segment_gpu_scapp(
    *,
    card: Any,
    segment_index: int,
    n_samples: int,
    data_i16: Any,
) -> None:
    import spcm
    from spcm_core import c_void_p, spcm_dwDefTransfer_i64

    card.set_i(spcm.SPC_SEQMODE_WRITESEGMENT, int(segment_index))
    card.set_i(spcm.SPC_SEQMODE_SEGMENTSIZE, int(n_samples))
    card._check_error(
        spcm_dwDefTransfer_i64(
            card._handle,
            spcm.SPCM_BUF_DATA,
            spcm.SPCM_DIR_GPUTOCARD,
            0,
            c_void_p(int(data_i16.data.ptr)),
            0,
            int(data_i16.nbytes),
        )
    )
    card.cmd(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)


def _set_sequence_graph_registers(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    card: Any,
) -> None:
    import spcm

    steps = repo.steps
    if not steps:
        raise ValueError("Cannot upload: compiled program has no steps")

    segment_lengths = _segment_lengths_from_repo(repo)
    max_segments = _segment_count_pow2(len(segment_lengths))
    card.set_i(spcm.SPC_SEQMODE_MAXSEGMENTS, int(max_segments))

    n_steps = len(steps)
    for step in steps:
        i = int(step.step_index)
        j = int(step.next_step)
        seg_idx = int(step.segment_index)
        loops = int(step.loops)
        if not (0 <= i < n_steps):
            raise ValueError(f"Invalid step_index={i} for step table size {n_steps}")
        if not (0 <= j < n_steps):
            raise ValueError(f"Invalid next_step={j} for step table size {n_steps}")
        if not (0 <= seg_idx < len(segment_lengths)):
            raise ValueError(
                f"Step {i} references invalid segment_index={seg_idx} "
                f"for {len(segment_lengths)} segments"
            )

        flags = (
            spcm.SPCSEQ_ENDLOOPONTRIG if bool(step.on_trig) else spcm.SPCSEQ_ENDLOOPALWAYS
        )
        entry = (flags & ~spcm.SPCSEQ_LOOPMASK) | (loops & spcm.SPCSEQ_LOOPMASK)
        entry <<= 32
        entry |= ((j << 16) & spcm.SPCSEQ_NEXTSTEPMASK) | (
            seg_idx & spcm.SPCSEQ_SEGMENTMASK
        )
        card.set_i(spcm.SPC_SEQMODE_STEPMEM0 + i, int(entry))

    card.set_i(spcm.SPC_SEQMODE_STARTSTEP, int(steps[0].step_index))


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


def _full_scapp_upload(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    card: Any,
) -> SCAPPUploadSession:
    import spcm

    if card is None:
        raise ValueError("Full SCAPP upload requires an open `card` handle")

    features = getattr(card, "_features", None)
    if features is not None and not bool(int(features) & int(spcm.SPCM_FEAT_SCAPP)):
        raise ValueError(
            "Card does not report SCAPP feature support. "
            "Use mode='cpu' or enable the SCAPP option."
        )

    n_channels = int(repo.physical_setup.N_ch)
    segment_lengths = _segment_lengths_from_repo(repo)
    compiled_segments = repo.segments
    if len(compiled_segments) != len(segment_lengths):
        raise ValueError(
            "Full upload requires all segments to be compiled. "
            "Compile all segments first."
        )

    _set_sequence_graph_registers(repo, card=card)

    for idx, seg in enumerate(compiled_segments):
        data_i16 = _as_cupy_i16_segment_data(
            seg.data_i16,
            n_channels=n_channels,
            n_samples=segment_lengths[idx],
        )
        _upload_segment_gpu_scapp(
            card=card,
            segment_index=idx,
            n_samples=segment_lengths[idx],
            data_i16=data_i16,
        )

    return SCAPPUploadSession(
        card=card,
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

    _validate_session_compat(
        repo,
        n_channels=session.n_channels,
        segment_lengths=session.segment_lengths,
        steps_signature=session.steps_signature,
    )
    indices = _resolve_upload_indices(repo, segment_indices=segment_indices)

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


def _update_scapp_segments_only(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    session: SCAPPUploadSession,
    segment_indices: Sequence[int] | None,
) -> SCAPPUploadSession:
    if session.card is None:
        raise ValueError("SCAPPUploadSession.card is None")

    _validate_session_compat(
        repo,
        n_channels=session.n_channels,
        segment_lengths=session.segment_lengths,
        steps_signature=session.steps_signature,
    )
    indices = _resolve_upload_indices(repo, segment_indices=segment_indices)

    for idx in indices:
        if not (0 <= idx < len(session.segment_lengths)):
            raise ValueError(
                f"segment_indices contains out-of-range index {idx} "
                f"for {len(session.segment_lengths)} segments"
            )

    for idx in indices:
        seg = repo.compiled_segment(idx)
        data_i16 = _as_cupy_i16_segment_data(
            seg.data_i16,
            n_channels=session.n_channels,
            n_samples=session.segment_lengths[idx],
        )
        _upload_segment_gpu_scapp(
            card=session.card,
            segment_index=idx,
            n_samples=session.segment_lengths[idx],
            data_i16=data_i16,
        )

    return session


def readback_sequence_segments_to_numpy(
    *,
    card: Any,
    n_channels: int,
    segment_lengths: Sequence[int],
    segment_indices: Sequence[int] | None = None,
    use_memory_test_mode: bool = True,
) -> tuple[tuple[int, np.ndarray], ...]:
    """
    Read sequence segment sample data back from card memory into NumPy buffers.

    Notes
    -----
    - The card should be in sequence mode.
    - Generation cards usually reject `CARDTOPC` transfers in normal generation mode.
      By default this helper enables `SPC_MEMTEST` for each readback transfer to permit
      reverse-direction access (as documented by Spectrum).
    - On some hardware/driver combinations card-memory readback may still be unsupported.
    - Returned arrays are shaped `(n_channels, n_samples)` with dtype `int16`.
    """
    import spcm
    from spcm_core import c_void_p, spcm_dwDefTransfer_i64

    n_ch = int(n_channels)
    lengths = tuple(int(n) for n in segment_lengths)
    if n_ch <= 0:
        raise ValueError(f"n_channels must be >= 1, got {n_ch}")
    if any(n <= 0 for n in lengths):
        raise ValueError(f"All segment lengths must be > 0, got {lengths}")

    if segment_indices is None:
        indices = list(range(len(lengths)))
    else:
        indices = sorted(set(int(i) for i in segment_indices))
        if not indices:
            raise ValueError("segment_indices must be non-empty when provided")

    out: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        if not (0 <= idx < len(lengths)):
            raise ValueError(
                f"segment_indices contains out-of-range index {idx} "
                f"for {len(lengths)} segments"
            )
        n_samples = lengths[idx]
        buf = np.empty((n_ch, n_samples), dtype=np.int16, order="F")

        card.set_i(spcm.SPC_SEQMODE_WRITESEGMENT, int(idx))
        card.set_i(spcm.SPC_SEQMODE_SEGMENTSIZE, int(n_samples))

        memtest_prev: int | None = None
        try:
            if use_memory_test_mode:
                memtest_prev = int(card.get_i(spcm.SPC_MEMTEST))
                if memtest_prev == 0:
                    card.set_i(spcm.SPC_MEMTEST, 1)

            card._check_error(
                spcm_dwDefTransfer_i64(
                    card._handle,
                    spcm.SPCM_BUF_DATA,
                    spcm.SPCM_DIR_CARDTOPC,
                    0,
                    c_void_p(int(buf.ctypes.data)),
                    0,
                    int(buf.nbytes),
                )
            )
            card.cmd(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)
        finally:
            if use_memory_test_mode and memtest_prev == 0:
                card.set_i(spcm.SPC_MEMTEST, 0)

        out.append((idx, np.array(buf, copy=True)))

    return tuple(out)


def upload_sequence_program(
    repo: QIRtoSamplesSegmentCompiler,
    *,
    mode: UploadMode = "cpu",
    card: Any = None,
    cpu_session: UploadSession | None = None,
    segment_indices: Sequence[int] | None = None,
    upload_steps: bool = True,
) -> UploadSession:
    """
    Upload compiled sequence slots to hardware.

    Parameters
    ----------
    repo:
        Compiled slot container (`QIRtoSamplesSegmentCompiler`).
    mode:
        - "cpu": host-buffer upload (NumPy). If buffers are CuPy, convert to NumPy.
        - "scapp": GPU-buffer upload via SCAPP/RDMA (CuPy buffers + SCAPP option).
        - "auto": select "scapp" if compiled buffers are GPU-resident, else "cpu".
    card:
        Open hardware card handle. Required for full CPU upload (`upload_steps=True`).
    cpu_session:
        Existing session from a prior full upload. Required for data-only updates
        (`upload_steps=False`). Type must match `mode`:
        `CPUUploadSession` for `"cpu"` and `SCAPPUploadSession` for `"scapp"`.
    segment_indices:
        Optional segment indices to update for data-only mode. If omitted in data-only
        mode, all currently compiled segments in `repo` are uploaded.
    upload_steps:
        - True (default): full upload of all segments + step graph.
          Requires all segments compiled.
        - False: data-only segment update using existing `cpu_session`.
          Footgun: use this only when segment lengths and step graph are unchanged.
          It rewrites sample data only.

    Returns
    -------
    UploadSession
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
        if not isinstance(cpu_session, CPUUploadSession):
            raise ValueError(
                "Data-only CPU upload requires CPUUploadSession from mode='cpu'"
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
    if upload_steps:
        if segment_indices is not None:
            raise ValueError("segment_indices is only valid when upload_steps=False")
        if card is None:
            raise ValueError("Full SCAPP upload requires an open `card` handle")
        return _full_scapp_upload(repo, card=card)

    if cpu_session is None:
        raise ValueError(
            "Data-only SCAPP upload requires session from a prior full upload"
        )
    if not isinstance(cpu_session, SCAPPUploadSession):
        raise ValueError(
            "Data-only SCAPP upload requires SCAPPUploadSession from mode='scapp'"
        )
    if card is not None and cpu_session.card is not card:
        raise ValueError("Session belongs to a different card object than `card`")
    return _update_scapp_segments_only(
        repo,
        session=cpu_session,
        segment_indices=segment_indices,
    )
