"""AWG sequence-mode program builder and compiler.

This package implements a small pipeline:
`AWGProgramBuilder` (intent) → `ResolvedIR` (integer-sample primitives) →
`QuantizedIR` (hardware-friendly segments) → `QIRtoSamplesSegmentCompiler` (slot-based int16 segments + step table).
"""

from .builder import AWGProgramBuilder
from .calibration import AODSin2Calib, AWGPhysicalSetupInfo, LinearPositionToFreqCalib
from .intent_ir import IntentIR
from .resolved_ir import ResolvedIR
from .resolve import resolve_intent_ir
from .synth_samples import (
    QIRtoSamplesSegmentCompiler,
    CompiledSegment,
    CompiledVoltageSegment,
    SequenceStep,
    quantize_voltage_buffer_to_int16,
)
from .quantize import (
    QuantizedIR,
    SegmentQuantizationInfo,
    format_samples_time,
    quantize_resolved_ir,
    quantum_samples,
)
from .types import ChannelMap
from .upload import (
    CPUUploadSession,
    SCAPPUploadSession,
    UploadSession,
    readback_sequence_segments_to_numpy,
    upload_sequence_program,
)
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
    "QIRtoSamplesSegmentCompiler",
    "CompiledSegment",
    "CompiledVoltageSegment",
    "SequenceStep",
    "quantize_voltage_buffer_to_int16",
    "upload_sequence_program",
    "CPUUploadSession",
    "SCAPPUploadSession",
    "UploadSession",
    "readback_sequence_segments_to_numpy",
    "SegmentQuantizationInfo",
    "format_samples_time",
    "quantize_resolved_ir",
    "quantum_samples",
    "ChannelMap",
    "minimise_crest_factor_phases",
    "schroeder_phases_rad",
]
