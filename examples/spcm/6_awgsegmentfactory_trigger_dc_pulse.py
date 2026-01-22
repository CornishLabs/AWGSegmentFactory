"""
Trigger-delay / jitter measurement helper:

Outputs a DC "low" level while waiting for an external trigger, then jumps to a
DC "high" level for a fixed pulse width, then returns to waiting.

Implementation detail:
- AWGSegmentFactory only synthesizes summed sines, but a sine at 0 Hz is a constant:
    y = amp * sin(phase)
  so choosing phase=pi/2 yields a positive DC level y=amp.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from awgsegmentfactory import (
    AWGProgramBuilder,
    ResolvedIR,
    compile_sequence_program,
    format_samples_time,
    quantize_resolved_ir,
)


def _build_dc_pulse_program(
    *,
    sample_rate_hz: float,
    low: float,
    high: float,
    pulse_s: float,
) -> ResolvedIR:
    phase = float(np.pi / 2.0)
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .define("dc_low", logical_channel="H", freqs=[0.0], amps=[low], phases=[phase])
        .define("dc_high", logical_channel="H", freqs=[0.0], amps=[high], phases=[phase])
    )

    # Segment 0: repeats until trigger. Keep the intended duration tiny; quantization
    # will clamp it to the minimum loopable segment length.
    b.segment("wait_low", mode="wait_trig")
    b.tones("H").use_def("dc_low")
    b.hold(time=1e-9)

    # Segment 1: a deterministic-width "pulse", then wraps back to wait_low.
    b.segment("pulse_high", mode="once")
    b.tones("H").use_def("dc_high")
    b.hold(time=pulse_s)

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def _print_quantization_report(compiled) -> None:
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


def _setup_spcm_sequence_from_compiled(sequence, compiled) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate-hz", type=float, default=625e6)
    ap.add_argument("--low", type=float, default=0.0, help="DC low level in [-1,1]")
    ap.add_argument("--high", type=float, default=0.9, help="DC high level in [-1,1]")
    ap.add_argument("--pulse-us", type=float, default=20.0, help="Pulse width (µs)")
    ap.add_argument(
        "--segment-quantum-us",
        type=float,
        default=None,
        help="Loopable-segment quantum (µs). Lower -> faster trigger response.",
    )
    ap.add_argument("--step-samples", type=int, default=64)
    args = ap.parse_args()

    sample_rate_hz = float(args.sample_rate_hz)
    pulse_s = float(args.pulse_us) * 1e-6
    logical_channel_to_hardware_channel = {"H": 0}

    # Make the wait_trig segment as short as hardware allows by default:
    # 1ch min is 384 samples, so choose quantum_s such that quantum_samples == 384.
    if args.segment_quantum_us is None:
        segment_quantum_s = 384.0 / sample_rate_hz
    else:
        segment_quantum_s = float(args.segment_quantum_us) * 1e-6

    ir = _build_dc_pulse_program(
        sample_rate_hz=sample_rate_hz,
        low=float(args.low),
        high=float(args.high),
        pulse_s=pulse_s,
    )
    q = quantize_resolved_ir(
        ir,
        logical_channel_to_hardware_channel=logical_channel_to_hardware_channel,
        segment_quantum_s=segment_quantum_s,
        step_samples=int(args.step_samples),
    )

    # If you don't have a card connected, use a safe "typical" int16 full-scale.
    full_scale_default = 32767
    compiled = compile_sequence_program(
        q,
        gain=1.0,
        clip=0.9,
        full_scale=full_scale_default,
    )

    print(f"compiled segments: {len(compiled.segments)} | steps: {len(compiled.steps)}")
    _print_quantization_report(compiled)
    print(
        "Notes:\n"
        "- EXT0 trigger transitions from 'wait_low' -> 'pulse_high'.\n"
        "- 'pulse_high' then wraps back to 'wait_low' automatically.\n"
    )

    # Optional: upload to a Spectrum card (requires Spectrum driver + spcm Python package).
    try:
        import spcm
        from spcm import units
    except Exception as exc:
        print(f"spcm not available (skipping upload): {exc}")
        return

    try:
        with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
            card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

            channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
            channels.enable(True)
            channels.output_load(50*units.ohm)
            channels.amp(1 * units.V)
            channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

            trigger = spcm.Trigger(card)
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.ext0_mode(spcm.SPC_TM_POS)
            trigger.ext0_level0(0.3 * units.V)
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.termination(1) #50 Ohm
            trigger.delay(0)

            clock = spcm.Clock(card)
            clock.mode(spcm.SPC_CM_INTPLL)
            clock.sample_rate(compiled.sample_rate_hz * units.Hz)
            clock.clock_output(False)

            # Compile again with the card's exact DAC scaling.
            full_scale = int(card.max_sample_value()) - 1
            compiled = compile_sequence_program(
                q,
                gain=1.0,
                clip=0.9,
                full_scale=full_scale,
            )

            sequence = spcm.Sequence(card)
            _setup_spcm_sequence_from_compiled(sequence, compiled)
            print("sequence written; starting card (Ctrl+C to stop)")

            card.timeout(0)
            card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            finally:
                card.stop()
    except spcm.SpcmException as exc:
        print(f"Could not open Spectrum card (skipping upload): {exc}")
        return


if __name__ == "__main__":
    main()

