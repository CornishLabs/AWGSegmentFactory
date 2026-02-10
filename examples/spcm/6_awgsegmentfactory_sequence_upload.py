"""
End-to-end example:

AWGProgramBuilder -> ResolvedIR -> QuantizedIR -> per-segment int16 samples
-> upload to a Spectrum AWG using Sequence Replay Mode.

This file is safe to run without a Spectrum driver installed: it will still
compile the waveform and print the quantization report, then skip upload.
"""

from __future__ import annotations

import time

from awgsegmentfactory import (
    AWGPhysicalSetupInfo,
    AWGProgramBuilder,
    ResolvedIR,
    compile_sequence_program,
    format_samples_time,
    quantize_resolved_ir,
)


def _build_demo_program(
    *, sample_rate_hz: float
) -> ResolvedIR:
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define("init_H", logical_channel="H", freqs=[1e5], amps=[0.3], phases="auto")
        .define("init_V", logical_channel="V", freqs=[2e5], amps=[0.3], phases="auto")
    )

    # 0) Wait for trigger, output stable tones (this segment loops until a trigger event)
    b.segment("wait_start", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=40e-6)

    b.segment("ramp_down", mode="once")
    b.tones("V").ramp_amp_to(
        amps=[0.1],
        time=50e-6,
        kind="exp",
        tau=20e-6,
    )

    # 1) A short chirp on H (one-shot segment)
    b.segment("chirp_H", mode="once")
    b.tones("H").move(df=+1e5, time=100e-6, idxs=[0], kind="linear")


    b.segment("wait_start_2", mode="wait_trig")
    b.hold(time=40e-6)

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
    sample_rate_hz = 625e6
    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})

    ir = _build_demo_program(sample_rate_hz=sample_rate_hz)
    q = quantize_resolved_ir(
        ir,
        segment_quantum_s=4e-6
    )

    # If you don't have a card connected, use a safe "typical" int16 full-scale.
    full_scale_default = 32767
    compiled = compile_sequence_program(
        q,
        physical_setup=physical_setup,
        gain=1.0,
        clip=0.9,
        full_scale=full_scale_default,
    )

    print(f"compiled segments: {len(compiled.segments)} | steps: {len(compiled.steps)}")
    _print_quantization_report(compiled)

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

            # Configure enabled channels (H->CH0, V->CH1)
            channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
            channels.enable(True)
            channels.output_load(units.highZ)
            channels.amp(1 * units.V)
            channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

            # Triggers: EXT0 ends wait_trig steps.
            trigger = spcm.Trigger(card)
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.ext0_mode(spcm.SPC_TM_POS)
            trigger.ext0_level0(0.5 * units.V)
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.termination(1)
            trigger.delay(320)
            # print(trigger.avail_delay_step())

            # Sample clock
            clock = spcm.Clock(card)
            clock.mode(spcm.SPC_CM_INTPLL)
            clock.sample_rate(compiled.sample_rate_hz * units.Hz)
            clock.clock_output(False)

            # Compile again with the card's exact DAC scaling.
            full_scale = int(card.max_sample_value()) - 1
            compiled = compile_sequence_program(
                q,
                physical_setup=physical_setup,
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
