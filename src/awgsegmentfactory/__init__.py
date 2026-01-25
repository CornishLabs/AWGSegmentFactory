"""AWG sequence-mode program builder and compiler.

This package implements a small pipeline:
`AWGProgramBuilder` (intent) → `ResolvedIR` (integer-sample primitives) →
`QuantizedIR` (hardware-friendly segments) → `CompiledSequenceProgram` (int16 samples + step table).
"""

from .builder import AWGProgramBuilder
from .calibration import LinearPositionToFreqCalib
from .ir import IntentIR
from .program_ir import ResolvedIR
from .resolve import resolve_intent_ir
from .sample_compile import CompiledSequenceProgram, compile_sequence_program
from .sequence_compile import (
    QuantizedIR,
    SegmentQuantizationInfo,
    format_samples_time,
    quantize_resolved_ir,
    quantum_samples,
)

__all__ = [
    "AWGProgramBuilder",
    "LinearPositionToFreqCalib",
    "IntentIR",
    "ResolvedIR",
    "resolve_intent_ir",
    "QuantizedIR",
    "CompiledSequenceProgram",
    "compile_sequence_program",
    "SegmentQuantizationInfo",
    "format_samples_time",
    "quantize_resolved_ir",
    "quantum_samples",
]
