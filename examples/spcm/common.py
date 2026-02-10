"""Shared helpers for Spectrum upload examples."""

from __future__ import annotations

from awgsegmentfactory import format_samples_time


def print_quantization_report(compiled) -> None:
    """Print per-segment quantization metadata for a compiled program."""
    fs = float(compiled.sample_rate_hz)
    if not compiled.quantization:
        return
    q = compiled.quantization[0].quantum_samples
    step = compiled.quantization[0].step_samples
    print(f"segment quantum: {format_samples_time(q, fs)} | step: {step} samples")
    for qi in compiled.quantization:
        o = format_samples_time(qi.original_samples, fs)
        n = format_samples_time(qi.quantized_samples, fs)
        print(
            f"- {qi.name}: {o} -> {n} | mode={qi.mode} loop={qi.loop} loopable={qi.loopable}"
        )


def setup_spcm_sequence_from_compiled(sequence, compiled) -> None:
    """Write compiled segments/steps into an `spcm.Sequence` object."""
    segments_hw = []
    for seg in compiled.segments:
        s = sequence.add_segment(seg.n_samples)
        s[:, :] = seg.data_i16
        segments_hw.append(s)

    steps_hw = []
    for step in compiled.steps:
        steps_hw.append(
            sequence.add_step(segments_hw[step.segment_index], loops=step.loops)
        )

    sequence.entry_step(steps_hw[0])

    for step in compiled.steps:
        steps_hw[step.step_index].set_transition(
            steps_hw[step.next_step], on_trig=step.on_trig
        )

    sequence.write_setup()

