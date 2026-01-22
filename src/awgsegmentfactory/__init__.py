from .builder import AWGProgramBuilder
from .calibration import LinearPositionToFreqCalib
from .program_ir import ProgramIR
from .sample_compile import CompiledSequenceProgram, compile_sequence_program
from .sequence_compile import (
    SegmentQuantizationInfo,
    format_samples_time,
    quantize_program_ir,
    quantum_samples,
)

__all__ = [
    "AWGProgramBuilder",
    "LinearPositionToFreqCalib",
    "ProgramIR",
    "CompiledSequenceProgram",
    "compile_sequence_program",
    "SegmentQuantizationInfo",
    "format_samples_time",
    "quantize_program_ir",
    "quantum_samples",
]


def __getattr__(name: str):
    # Backwards-compatible convenience imports (debug helpers moved to `awgsegmentfactory.debug`).
    if name in {"interactive_grid_debug", "LinearFreqToPos", "sequence_samples_debug"}:
        try:
            from . import debug as _debug
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                f"`awgsegmentfactory.{name}` requires optional debug dependencies. "
                "Install the `dev` dependency group (matplotlib, ipywidgets, etc.)."
            ) from exc
        return getattr(_debug, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
