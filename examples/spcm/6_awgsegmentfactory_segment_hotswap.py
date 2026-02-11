"""
Hot-swap one segment's data in Spectrum sequence mode.

Flow:
1) Build + upload a full sequence once.
2) Recompile with a different amplitude for one segment.
3) Upload only that segment's sample data (step graph unchanged).
"""

from __future__ import annotations

import argparse
import math
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


HOTSWAP_SEGMENT_INDEX = 1  # "pulse" segment in this example
HOTSWAP_LOGICAL_CHANNEL = "H"


def _build_hotswap_program(
    *,
    sample_rate_hz: float,
    low_mv: float,
    pulse_mv: float,
    pulse_s: float,
) -> ResolvedIR:
    phase = float(math.pi / 2.0)  # 0 Hz tone => DC level = amp * sin(phase)
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .define("dc_low", logical_channel="H", freqs=[0.0], amps=[low_mv], phases=[phase])
        .define("dc_pulse", logical_channel="H", freqs=[0.0], amps=[pulse_mv], phases=[phase])
    )

    b.segment("wait_low", mode="wait_trig")
    b.tones("H").use_def("dc_low")
    b.hold(time=20e-6)

    b.segment("pulse", mode="once")
    b.tones("H").use_def("dc_pulse")
    b.hold(time=pulse_s)

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def _set_pulse_amplitude_in_quantized(
    *,
    quantized,
    pulse_mv: float,
) -> None:
    """Mutate the pulse segment amplitude in-place inside a QuantizedIR object."""
    seg = quantized.resolved_ir.segments[HOTSWAP_SEGMENT_INDEX]
    for part in seg.parts:
        pp = part.logical_channels[HOTSWAP_LOGICAL_CHANNEL]
        pp.start.amps[...] = float(pulse_mv)
        pp.end.amps[...] = float(pulse_mv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate-hz", type=float, default=625e6)
    ap.add_argument("--full-scale-mv", type=float, default=1000.0)
    ap.add_argument("--low-mv", type=float, default=100.0)
    ap.add_argument("--pulse-us", type=float, default=40.0)
    ap.add_argument("--segment-quantum-us", type=float, default=4.0)
    ap.add_argument(
        "--pulse-values-mv",
        type=float,
        nargs="+",
        default=[250.0, 450.0, 700.0, 900.0],
        help="Pulse amplitudes (mV) to hot-swap into segment index 1.",
    )
    ap.add_argument("--update-period-s", type=float, default=1.0)
    args = ap.parse_args()

    sample_rate_hz = float(args.sample_rate_hz)
    full_scale_mv = float(args.full_scale_mv)
    low_mv = float(args.low_mv)
    pulse_s = float(args.pulse_us) * 1e-6
    segment_quantum_s = float(args.segment_quantum_us) * 1e-6
    pulse_values_mv = [float(v) for v in args.pulse_values_mv]
    update_period_s = float(args.update_period_s)
    if len(pulse_values_mv) < 1:
        raise ValueError("--pulse-values-mv must contain at least one value")

    import spcm
    from spcm import units

    with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
        card.stop()
        card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(full_scale_mv * units.mV)
        channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_EXT0)
        trigger.ext0_mode(spcm.SPC_TM_POS)
        trigger.ext0_level0(0.3 * units.V)
        trigger.ext0_coupling(spcm.COUPLING_DC)
        trigger.termination(1)
        trigger.delay(0)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(sample_rate_hz * units.Hz)
        clock.clock_output(False)

        full_scale = int(card.max_sample_value()) - 1

        ir = _build_hotswap_program(
            sample_rate_hz=sample_rate_hz,
            low_mv=low_mv,
            pulse_mv=pulse_values_mv[0],
            pulse_s=pulse_s,
        )
        q = quantize_resolved_ir(ir, segment_quantum_s=segment_quantum_s, step_samples=64)
        setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0})
        slots_compiler = QIRtoSamplesSegmentCompiler.initialise_from_quantised(
            quantized=q,
            physical_setup=setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        )
        slots_compiler.compile()
        session = upload_sequence_program(slots_compiler, mode="cpu", card=card, upload_steps=True)
        print("initial full upload complete")
        print_quantization_report(slots_compiler)

        card.timeout(0)
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)
        print("card started")

        # Data-only hot-swap updates for one segment.
        try:
            for pulse_mv in pulse_values_mv[1:]:
                _set_pulse_amplitude_in_quantized(
                    quantized=q,
                    pulse_mv=pulse_mv,
                )
                slots_compiler.compile(segment_indices=[HOTSWAP_SEGMENT_INDEX])
                session = upload_sequence_program(
                    slots_compiler,
                    mode="cpu",
                    card=card,
                    cpu_session=session,
                    segment_indices=[HOTSWAP_SEGMENT_INDEX],
                    upload_steps=False,
                )
                print(
                    f"hot-swapped segment {HOTSWAP_SEGMENT_INDEX} with pulse_mv={pulse_mv:.1f}"
                )
                time.sleep(update_period_s)

            print("update list exhausted; running until Ctrl+C")
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            card.stop()


if __name__ == "__main__":
    main()
