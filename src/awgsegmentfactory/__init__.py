"""AWG sequence-mode program builder and compiler.

This package implements a small pipeline:
`AWGProgramBuilder` (intent) → `ResolvedIR` (integer-sample primitives) →
`QuantizedIR` (hardware-friendly segments) → `CompiledSequenceProgram` (int16 samples + step table).
"""

from .builder import AWGProgramBuilder
from .calibration import AODTanh2Calib, LinearPositionToFreqCalib
from .intent_ir import IntentIR
from .resolved_ir import ResolvedIR
from .resolve import resolve_intent_ir
from .synth_samples import (
    CompiledSequenceProgram,
    compile_sequence_program,
    compiled_sequence_program_to_numpy,
)
from .quantize import (
    QuantizedIR,
    SegmentQuantizationInfo,
    format_samples_time,
    quantize_resolved_ir,
    quantum_samples,
)
from .types import ChannelMap
from .upload import upload_sequence_program
from .phase_minimiser import minimise_crest_factor_phases, schroeder_phases_rad

__all__ = [
    "AWGProgramBuilder",
    "LinearPositionToFreqCalib",
    "AODTanh2Calib",
    "IntentIR",
    "ResolvedIR",
    "resolve_intent_ir",
    "QuantizedIR",
    "CompiledSequenceProgram",
    "compile_sequence_program",
    "compiled_sequence_program_to_numpy",
    "upload_sequence_program",
    "SegmentQuantizationInfo",
    "format_samples_time",
    "quantize_resolved_ir",
    "quantum_samples",
    "ChannelMap",
    "minimise_crest_factor_phases",
    "schroeder_phases_rad",
]
