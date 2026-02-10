"""AWG sequence-mode program builder and compiler.

This package implements a small pipeline:
`AWGProgramBuilder` (intent) → `ResolvedIR` (integer-sample primitives) →
`QuantizedIR` (hardware-friendly segments) → `CompiledSequenceProgram` (int16 samples + step table).
"""

from .builder import AWGProgramBuilder
from .calibration import AODSin2Calib, AWGPhysicalSetupInfo, LinearPositionToFreqCalib
from .intent_ir import IntentIR
from .resolved_ir import ResolvedIR
from .resolve import resolve_intent_ir
from .synth_samples import (
    CompiledSequenceProgram,
    SynthesizedSequenceProgram,
    compile_sequence_program,
    compiled_sequence_program_to_numpy,
    synthesize_sequence_program,
    synthesized_sequence_program_to_numpy,
    quantise_and_normalise_voltage_for_awg,
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
    "AODSin2Calib",
    "AWGPhysicalSetupInfo",
    "IntentIR",
    "ResolvedIR",
    "resolve_intent_ir",
    "QuantizedIR",
    "CompiledSequenceProgram",
    "SynthesizedSequenceProgram",
    "compile_sequence_program",
    "synthesize_sequence_program",
    "quantise_and_normalise_voltage_for_awg",
    "compiled_sequence_program_to_numpy",
    "synthesized_sequence_program_to_numpy",
    "upload_sequence_program",
    "SegmentQuantizationInfo",
    "format_samples_time",
    "quantize_resolved_ir",
    "quantum_samples",
    "ChannelMap",
    "minimise_crest_factor_phases",
    "schroeder_phases_rad",
]
