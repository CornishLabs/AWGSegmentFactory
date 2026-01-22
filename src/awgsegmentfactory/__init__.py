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
    "interactive_grid_debug",
    "LinearFreqToPos",
]


def __getattr__(name: str):
    if name in {"interactive_grid_debug", "LinearFreqToPos"}:
        try:
            from .debug_plot import interactive_grid_debug, LinearFreqToPos
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                f"`awgsegmentfactory.{name}` requires the optional debug-plot dependencies. "
                "Install the `dev` dependency group (matplotlib, ipywidgets, etc.)."
            ) from exc
        return interactive_grid_debug if name == "interactive_grid_debug" else LinearFreqToPos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
