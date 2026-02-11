"""
Hot-swap one segment's data in Spectrum sequence mode.

Flow:
1) Build + upload a full sequence once.
2) On keyboard command, recompile one segment with a new random DC ramp.
3) Upload only that segment's sample data (step graph unchanged).
4) After a short delay, issue a software trigger.

Hot-swap safety note:
- This demo keeps the hot-swapped segment in `phase_mode="manual"` so partial
  recompile is local and deterministic.
- For general multitone sequences with `phase_mode="continue"`, recompiling one
  segment can invalidate predecessor/successor continuity assumptions. Recompile
  a contiguous suffix when in doubt.
"""

from __future__ import annotations

import argparse
import math
import random
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager

from awgsegmentfactory import (
    QIRtoSamplesSegmentCompiler,
    AWGPhysicalSetupInfo,
    AWGProgramBuilder,
    ResolvedIR,
    quantize_resolved_ir,
    upload_sequence_program,
)

from common import print_quantization_report


HOTSWAP_SEGMENT_INDEX = 1  # "ramp" segment in this example
HOTSWAP_LOGICAL_CHANNEL = "H"


def _build_hotswap_program(
    *,
    sample_rate_hz: float,
    low_mv: float,
    ramp_start_mv: float,
    ramp_end_mv: float,
    ramp_s: float,
) -> ResolvedIR:
    phase = float(math.pi / 2.0)  # 0 Hz tone => DC level = amp * sin(phase)
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .define("dc_low", logical_channel="H", freqs=[0.0], amps=[low_mv], phases=[phase])
        .define(
            "dc_ramp_start",
            logical_channel="H",
            freqs=[0.0],
            amps=[ramp_start_mv],
            phases=[phase],
        )
    )

    b.segment("wait_low", mode="wait_trig")
    b.tones("H").use_def("dc_low")
    b.hold(time=20e-6)

    # Keep this segment manual so one-segment recompile does not depend on predecessor phase state.
    b.segment("ramp", mode="once", phase_mode="manual")
    b.tones("H").use_def("dc_ramp_start")
    b.tones("H").ramp_amp_to(
        amps=[ramp_end_mv],
        time=ramp_s,
        kind="linear",
    )

    return b.build_resolved_ir(sample_rate_hz=sample_rate_hz)


def _set_ramp_amplitudes_in_quantised(
    *,
    quantised,
    ramp_start_mv: float,
    ramp_end_mv: float,
) -> None:
    """
    Mutate the ramp segment amplitudes in-place inside a QuantizedIR object.

    The first part is set to start->end. Any trailing quantization hold-parts are set
    to end->end to preserve continuity within the segment.
    """
    seg = quantised.resolved_ir.segments[HOTSWAP_SEGMENT_INDEX]
    if not seg.parts:
        raise RuntimeError("Expected ramp segment to contain at least one part")

    first = seg.parts[0]
    pp0 = first.logical_channels[HOTSWAP_LOGICAL_CHANNEL]
    pp0.start.amps[...] = float(ramp_start_mv)
    pp0.end.amps[...] = float(ramp_end_mv)

    for part in seg.parts[1:]:
        pp = part.logical_channels[HOTSWAP_LOGICAL_CHANNEL]
        pp.start.amps[...] = float(ramp_end_mv)
        pp.end.amps[...] = float(ramp_end_mv)


def _random_ramp_norm(rng: random.Random) -> tuple[float, float]:
    """Return `(start_norm, end_norm)` in [0, 1)."""
    return rng.random(), rng.random()


@contextmanager
def _linux_cbreak_stdin():
    """
    Put stdin into cbreak mode for single-key commands on Linux terminals.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_nonblocking(timeout_s: float) -> str | None:
    """
    Return one key from stdin or None if no key is available before timeout.
    """
    ready, _, _ = select.select([sys.stdin], [], [], float(timeout_s))
    if not ready:
        return None
    return sys.stdin.read(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate-hz", type=float, default=625e6)
    ap.add_argument("--full-scale-mv", type=float, default=1000.0)
    ap.add_argument("--low-mv", type=float, default=100.0)
    ap.add_argument("--ramp-us", type=float, default=40.0)
    ap.add_argument("--segment-quantum-us", type=float, default=4.0)
    ap.add_argument(
        "--pre-trigger-ms",
        type=float,
        default=3.0,
        help="Delay between hotswap upload and software trigger (ms).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random ramps.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sample_rate_hz = float(args.sample_rate_hz)
    full_scale_mv = float(args.full_scale_mv)
    low_mv = float(args.low_mv)
    ramp_s = float(args.ramp_us) * 1e-6
    segment_quantum_s = float(args.segment_quantum_us) * 1e-6
    pre_trigger_delay_s = float(args.pre_trigger_ms) * 1e-3

    start_norm0, end_norm0 = _random_ramp_norm(rng)
    start_mv0 = start_norm0 * full_scale_mv
    end_mv0 = end_norm0 * full_scale_mv

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

        # Software trigger mode (no external trigger line).
        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_NONE)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(sample_rate_hz * units.Hz)
        clock.clock_output(False)

        full_scale = int(card.max_sample_value()) - 1

        ir = _build_hotswap_program(
            sample_rate_hz=sample_rate_hz,
            low_mv=low_mv,
            ramp_start_mv=start_mv0,
            ramp_end_mv=end_mv0,
            ramp_s=ramp_s,
        )
        q = quantize_resolved_ir(ir, segment_quantum_s=segment_quantum_s, step_samples=64)
        setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0})
        slots_compiler = QIRtoSamplesSegmentCompiler(
            quantised=q,
            physical_setup=setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        )
        slots_compiler.compile_to_card_int16()
        session = upload_sequence_program(slots_compiler, mode="cpu", card=card, upload_steps=True)
        print("initial full upload complete")
        print_quantization_report(slots_compiler)

        card.timeout(0)
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
        print("card started")
        print(
            "Controls:\n"
            "  t / space / Enter : hotswap random ramp then software trigger\n"
            "  q or ESC           : quit\n"
        )
        print(
            f"initial ramp: {start_norm0:.3f} -> {end_norm0:.3f} "
            f"(norm), {start_mv0:.1f} -> {end_mv0:.1f} mV"
        )

        # Data-only hot-swap + software trigger on user command.
        try:
            with _linux_cbreak_stdin():
                while True:
                    key = _read_key_nonblocking(timeout_s=0.1)
                    if key is None:
                        continue
                    if key in ("\x1b", "q", "Q"):  # ESC / q
                        break
                    if key not in ("t", "T", " ", "\n", "\r"):
                        continue

                    start_norm, end_norm = _random_ramp_norm(rng)
                    start_mv = start_norm * full_scale_mv
                    end_mv = end_norm * full_scale_mv

                    _set_ramp_amplitudes_in_quantised(
                        quantised=q,
                        ramp_start_mv=start_mv,
                        ramp_end_mv=end_mv,
                    )
                    slots_compiler.compile_to_card_int16(
                        segment_indices=[HOTSWAP_SEGMENT_INDEX]
                    )
                    session = upload_sequence_program(
                        slots_compiler,
                        mode="cpu",
                        card=card,
                        cpu_session=session,
                        segment_indices=[HOTSWAP_SEGMENT_INDEX],
                        upload_steps=False,
                    )
                    if pre_trigger_delay_s > 0:
                        time.sleep(pre_trigger_delay_s)
                    trigger.force()
                    print(
                        f"triggered ramp {start_norm:.3f}->{end_norm:.3f} (norm), "
                        f"{start_mv:.1f}->{end_mv:.1f} mV "
                        f"(segment {HOTSWAP_SEGMENT_INDEX})"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            card.stop()


if __name__ == "__main__":
    main()
