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
    QIRtoSamplesSegmentCompiler,
    AWGPhysicalSetupInfo,
    AWGProgramBuilder,
    ResolvedIR,
    quantize_resolved_ir,
    upload_sequence_program,
)

from common import print_quantization_report


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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate-hz", type=float, default=625e6)
    ap.add_argument("--low", type=float, default=0.0, help="DC low level in mV")
    ap.add_argument("--high", type=float, default=900.0, help="DC high level in mV")
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
    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0})

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
        segment_quantum_s=segment_quantum_s,
        step_samples=int(args.step_samples),
    )

    # Match channels.amp(1 * units.V): full-scale output is 1000 mV.
    full_scale_mv = 1000.0
    import spcm
    from spcm import units

    with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
        card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(1 * units.V)
        channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_EXT0)
        trigger.ext0_mode(spcm.SPC_TM_POS)
        trigger.ext0_level0(0.3 * units.V)
        trigger.ext0_coupling(spcm.COUPLING_DC)
        trigger.termination(1)  # 50 Ohm
        trigger.delay(0)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(sample_rate_hz * units.Hz)
        clock.clock_output(False)

        # Quantize once with the card's exact DAC scaling.
        full_scale = int(card.max_sample_value()) - 1
        slots_compiler = QIRtoSamplesSegmentCompiler(
            quantised=q,
            physical_setup=physical_setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        )
        slots_compiler.compile_to_card_int16()

        print(f"compiled segments: {len(slots_compiler.segments)} | steps: {len(slots_compiler.steps)}")
        print_quantization_report(slots_compiler)
        print(
            "Notes:\n"
            "- EXT0 trigger transitions from 'wait_low' -> 'pulse_high'.\n"
            "- 'pulse_high' then wraps back to 'wait_low' automatically.\n"
        )

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
