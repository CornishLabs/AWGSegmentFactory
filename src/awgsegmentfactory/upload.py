"""Upload helpers (compiled segments -> hardware).

This module intentionally keeps the public API stable while hardware backends evolve.

Today:
- "cpu" upload is typically done using Spectrum's normal DMA APIs (host buffers).

Future:
- "scapp" upload will target Spectrum SCAPP / RDMA (GPU buffers).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .synth_samples import CompiledSequenceProgram, compiled_sequence_program_to_numpy


UploadMode = Literal["cpu", "scapp", "auto"]


def upload_sequence_program(
    prog: CompiledSequenceProgram,
    *,
    mode: UploadMode = "cpu",
):
    """
    Upload a compiled sequence program to hardware.

    Parameters
    ----------
    prog:
        Output of `compile_sequence_program(...)`. Segment buffers may be NumPy or CuPy
        depending on `output=...`.
    mode:
        - "cpu": host-buffer upload (NumPy). If buffers are CuPy, this will convert
          to NumPy first.
        - "scapp": GPU-buffer upload via SCAPP/RDMA (requires CuPy buffers).
        - "auto": select "scapp" if buffers are CuPy, else "cpu".
    """
    if mode not in ("cpu", "scapp", "auto"):
        raise ValueError("mode must be 'cpu', 'scapp', or 'auto'")

    first = prog.segments[0].data_i16 if prog.segments else np.zeros((0, 0), np.int16)
    is_numpy = isinstance(first, np.ndarray)

    if mode == "auto":
        mode = "cpu" if is_numpy else "scapp"

    if mode == "cpu":
        prog_cpu = compiled_sequence_program_to_numpy(prog)
        raise NotImplementedError(
            "CPU upload is not implemented as a library function yet. "
            "See examples/spcm/6_awgsegmentfactory_sequence_upload.py for a working reference."
        )

    # mode == "scapp"
    raise NotImplementedError(
        "SCAPP/RDMA upload for sequence-mode segments is not implemented yet."
    )

