import unittest

import numpy as np

from awgsegmentfactory.ir import (
    DefinitionSpec,
    HoldOp,
    MoveOp,
    ProgramSpec,
    RampAmpToOp,
    SegmentSpec,
    UseDefOp,
)
from awgsegmentfactory.resolve import resolve_program_ir


class TestProgramIR(unittest.TestCase):
    def test_program_ir_roundtrips_to_timeline(self) -> None:
        fs = 4.0  # dt=0.25s (exact), makes time comparisons stable.

        spec = ProgramSpec(
            sample_rate_hz=fs,
            logical_channels=("H", "V"),
            definitions={
                "dH": DefinitionSpec(
                    name="dH",
                    logical_channel="H",
                    freqs_hz=(100.0,),
                    amps=(1.0,),
                    phases_rad=(0.0,),
                ),
                "dV": DefinitionSpec(
                    name="dV",
                    logical_channel="V",
                    freqs_hz=(200.0,),
                    amps=(1.0,),
                    phases_rad=(0.0,),
                ),
            },
            segments=(
                SegmentSpec(
                    name="init",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        UseDefOp(logical_channel="H", def_name="dH"),
                        UseDefOp(logical_channel="V", def_name="dV"),
                        HoldOp(time_s=1.0),
                    ),
                ),
                SegmentSpec(
                    name="move_row_up",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        MoveOp(
                            logical_channel="V", df_hz=2.0, time_s=1.0, idxs=(0,)
                        ),
                    ),
                ),
                SegmentSpec(
                    name="wait_for_trigger_B",
                    mode="wait_trig",
                    loop=1,
                    ops=(HoldOp(time_s=1.0),),
                ),
                SegmentSpec(
                    name="ramp_off",
                    mode="loop_n",
                    loop=1,
                    ops=(
                        RampAmpToOp(logical_channel="H", amps_target=0.0, time_s=1.0),
                        RampAmpToOp(logical_channel="V", amps_target=0.0, time_s=1.0),
                    ),
                ),
                SegmentSpec(
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

        ir = resolve_program_ir(spec)
        self.assertEqual([s.name for s in ir.segments], [s.name for s in spec.segments])
        self.assertEqual(ir.n_samples, 24)

        tl = ir.to_timeline()
        self.assertEqual(tl.t_end, 6.0)

        sH_while_V_moves = tl.state_at("H", 1.5)
        np.testing.assert_allclose(sH_while_V_moves.freqs_hz, [100.0])
        np.testing.assert_allclose(sH_while_V_moves.amps, [1.0])

        sV_while_H_ramps = tl.state_at("V", 3.5)
        np.testing.assert_allclose(sV_while_H_ramps.freqs_hz, [202.0])
        np.testing.assert_allclose(sV_while_H_ramps.amps, [1.0])
