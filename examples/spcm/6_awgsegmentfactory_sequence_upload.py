"""
End-to-end example:

AWGProgramBuilder -> ResolvedIR -> QuantizedIR -> per-segment int16 samples
-> upload to a Spectrum AWG using Sequence Replay Mode.
"""

from __future__ import annotations

import time

from awgsegmentfactory import (
    QIRtoSamplesSegmentCompiler,
    AWGPhysicalSetupInfo,
    AWGProgramBuilder,
    ResolvedIR,
    quantize_resolved_ir,
    upload_sequence_program,
)

from common import print_quantization_report


def _build_demo_program(
    *, sample_rate_hz: float
) -> ResolvedIR:
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        # Uncalibrated channels: amplitudes are RF voltages in mV.
        .define("init_H", logical_channel="H", freqs=[1e5], amps=[300.0], phases="auto")
        .define("init_V", logical_channel="V", freqs=[2e5], amps=[250.0], phases="auto")
    )

    # 0) Wait for trigger, output stable tones (this segment loops until a trigger event)
    b.segment("wait_start", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=40e-6)

    b.segment("ramp_down", mode="once")
    b.tones("V").ramp_amp_to(
        amps=[100.0],
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

def main() -> None:
    sample_rate_hz = 625e6

    ir = _build_demo_program(sample_rate_hz=sample_rate_hz)
    q = quantize_resolved_ir(
        ir,
        segment_quantum_s=4e-6
    )

    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})
    full_scale_mv = 1000.0

    # Match channels.amp(1 * units.V): full-scale output is 1000 mV.
    import spcm
    from spcm import units

    with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
        card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

        # Configure enabled channels (H->CH0, V->CH1)
        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
        channels.enable(True)
        channels.output_load(50*units.ohm)
        channels.amp(full_scale_mv * units.mV)
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
        clock.sample_rate(sample_rate_hz * units.Hz)
        clock.clock_output(False)

        # Quantize once with the card's exact DAC scaling.
        full_scale = int(card.max_sample_value()) - 1
        slots_compiler = QIRtoSamplesSegmentCompiler.initialise_from_quantised(
            quantised=q,
            physical_setup=physical_setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        )
        slots_compiler.compile()

        print(f"compiled segments: {len(slots_compiler.segments)} | steps: {len(slots_compiler.steps)}")
        print_quantization_report(slots_compiler)

        _session = upload_sequence_program(slots_compiler, mode="cpu", card=card)
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


if __name__ == "__main__":
    main()
