"""
Minimal Spectrum sequence-mode trigger test (no AWGSegmentFactory).

Two segments on CH0:
- `seg_low` repeats until an EXT0 trigger occurs.
- transition to `seg_high` (one pulse), then return to `seg_low`.

Notes:
- Segment length constraints depend on the card. Typical sequence mode constraints are:
  minimum segment length ~384 samples (1 channel) and sizes multiple-of ~32 samples.
- Trigger-to-output-change latency is quantized to the currently-playing segment length.
"""

from __future__ import annotations

import time

import numpy as np
import spcm
from spcm import units


SAMPLE_RATE_HZ = 625e6
WAIT_SAMPLES = 32 * 12
PULSE_SAMPLES = 3200
LOW = 0.0
HIGH = 0.6


def main() -> None:
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
        trigger.ext0_level0(0.2 * units.V)
        trigger.ext0_coupling(spcm.COUPLING_DC)
        trigger.termination(1)
        trigger.delay(0)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(max=True)
        clock.clock_output(False)

        full_scale = int(card.max_sample_value()) - 1
        low_i16 = np.int16(int(round(LOW * full_scale)))
        high_i16 = np.int16(int(round(HIGH * full_scale)))

        sequence = spcm.Sequence(card)
        seg_low = sequence.add_segment(WAIT_SAMPLES)  # [channel, sample]
        seg_high = sequence.add_segment(PULSE_SAMPLES)
        seg_low[:, :] = low_i16
        seg_high[:, :] = high_i16

        step0 = sequence.add_step(seg_low, loops=1)
        step1 = sequence.add_step(seg_high, loops=1)
        sequence.entry_step(step0)
        step0.set_transition(step1, on_trig=True)
        step1.set_transition(step0, on_trig=False)
        sequence.write_setup()

        print(sequence)
        print(
            f"wait={WAIT_SAMPLES} ({WAIT_SAMPLES / SAMPLE_RATE_HZ * 1e6:.3f} µs) | "
            f"pulse={PULSE_SAMPLES} ({PULSE_SAMPLES / SAMPLE_RATE_HZ * 1e6:.3f} µs)"
        )

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
