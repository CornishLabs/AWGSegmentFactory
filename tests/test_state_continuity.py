import unittest

import numpy as np

from awgsegmentfactory.intent_ir import (
    IntentDefinition,
    HoldOp,
    MoveOp,
    IntentIR,
    RampAmpToOp,
    IntentSegment,
    UseDefOp,
)
from awgsegmentfactory.resolve import resolve_intent_ir


class TestStateContinuity(unittest.TestCase):
    def test_spans_are_continuous_and_state_propagates(self) -> None:
        fs = 4.0  # dt=0.25s (exact), makes time comparisons stable.

        intent = IntentIR(
            logical_channels=("H", "V"),
            definitions={
                "dH": IntentDefinition(
                    name="dH",
                    logical_channel="H",
                    freqs_hz=(100.0,),
                    amps=(1.0,),
                    phases_rad=(0.0,),
                ),
                "dV": IntentDefinition(
                    name="dV",
                    logical_channel="V",
                    freqs_hz=(200.0,),
                    amps=(1.0,),
                    phases_rad=(0.0,),
                ),
            },
            segments=(
                IntentSegment(
                    name="init",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        UseDefOp(logical_channel="H", def_name="dH"),
                        UseDefOp(logical_channel="V", def_name="dV"),
                        HoldOp(time_s=1.0),
                    ),
                ),
                IntentSegment(
                    name="move_row_up",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        MoveOp(
                            logical_channel="V", df_hz=2.0, time_s=1.0, idxs=(0,)
                        ),
                    ),
                ),
                IntentSegment(
                    name="wait_for_trigger_B",
                    mode="wait_trig",
                    loop=1,
                    ops=(HoldOp(time_s=1.0),),
                ),
                IntentSegment(
                    name="ramp_off",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        RampAmpToOp(logical_channel="H", amps_target=0.0, time_s=1.0),
                        RampAmpToOp(logical_channel="V", amps_target=0.0, time_s=1.0),
                    ),
                ),
                IntentSegment(
                    name="move_back",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        MoveOp(
                            logical_channel="V", df_hz=-2.0, time_s=1.0, idxs=(0,)
                        ),
                    ),
                ),
            ),
            calibrations={},
        )

        tl = resolve_intent_ir(intent, sample_rate_hz=fs).to_timeline()

        # State continuity checks inside intervals where (previously) the logical channel had no spans.
        sH_while_V_moves = tl.state_at("H", 1.5)
        np.testing.assert_allclose(sH_while_V_moves.freqs_hz, [100.0])
        np.testing.assert_allclose(sH_while_V_moves.amps, [1.0])

        sV_while_H_ramps = tl.state_at("V", 3.5)
        np.testing.assert_allclose(sV_while_H_ramps.freqs_hz, [202.0])
        np.testing.assert_allclose(sV_while_H_ramps.amps, [1.0])

        # Structural check: every logical-channel span list covers the full timeline with no gaps.
        for logical_channel in ("H", "V"):
            spans = tl.spans_by_logical_channel[logical_channel]
            self.assertEqual(spans[0].t0, 0.0)
            for prev, cur in zip(spans, spans[1:]):
                self.assertEqual(prev.t1, cur.t0)
            self.assertEqual(spans[-1].t1, tl.t_end)
