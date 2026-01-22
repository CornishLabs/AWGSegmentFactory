import unittest

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.ir import HoldOp, RemapFromDefOp, UseDefOp
from awgsegmentfactory.sequence_compile import quantize_program_ir


class TestBuilder(unittest.TestCase):
    def test_build_ir_basic_hold_is_quantized(self) -> None:
        fs = 4.0  # dt=0.25s
        b = (
            AWGProgramBuilder(sample_rate=fs)
            .plane("H")
            .define("dH", plane="H", freqs=[100.0], amps=[1.0], phases="auto")
            .segment("seg0", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=0.3)
        )

        ir = b.build_ir()
        self.assertEqual(len(ir.segments), 1)
        self.assertEqual(ir.segments[0].name, "seg0")
        self.assertEqual(ir.segments[0].mode, "loop_n")  # "once" becomes loop_n, loop=1
        self.assertEqual(ir.segments[0].loop, 1)
        self.assertEqual(ir.segments[0].n_samples, 2)  # ceil(0.3 * 4) = 2

        part = ir.segments[0].parts[0]
        self.assertEqual(part.n_samples, 2)
        np.testing.assert_allclose(part.planes["H"].start.freqs_hz, [100.0])
        np.testing.assert_allclose(part.planes["H"].start.amps, [1.0])

        # build() returns the debug timeline view; it should match the IR duration.
        tl = b.build()
        self.assertEqual(tl.t_end, ir.duration_s)

    def test_build_spec_records_ops_and_expands_remap_dst_all(self) -> None:
        b = AWGProgramBuilder(sample_rate=10.0).plane("H")
        b.define(
            "start",
            plane="H",
            freqs=[1.0, 2.0, 3.0],
            amps=[1.0, 1.0, 1.0],
            phases="auto",
        )
        b.define(
            "target",
            plane="H",
            freqs=[4.0, 5.0, 6.0],
            amps=[1.0, 1.0, 1.0],
            phases="auto",
        )

        b.segment("init", mode="once")
        b.tones("H").use_def("start")
        b.hold(time=0.1)

        b.segment("remap", mode="once")
        b.tones("H").remap_from_def(
            target_def="target", src=[0, 1, 2], dst="all", time=0.1
        )

        spec = b.build_spec()
        self.assertEqual([s.name for s in spec.segments], ["init", "remap"])

        ops0 = spec.segments[0].ops
        self.assertIsInstance(ops0[0], UseDefOp)
        self.assertIsInstance(ops0[1], HoldOp)

        ops1 = spec.segments[1].ops
        self.assertIsInstance(ops1[0], RemapFromDefOp)
        self.assertEqual(ops1[0].dst, (0, 1, 2))

    def test_wait_trig_hold_snaps_in_quantize_stage(self) -> None:
        fs = 1000.0
        f0 = 7.0
        hold_s = 0.11  # 110 samples

        b = (
            AWGProgramBuilder(sample_rate=fs)
            .plane("H")
            .plane("V")
            .plane("A")
            .plane("B")
            .define("dH", plane="H", freqs=[f0], amps=[1.0], phases="auto")
            .segment("wait", mode="wait_trig")
            .tones("H")
            .use_def("dH")
            .hold(time=hold_s, warn_df=0.1)
        )

        ir = b.build_ir()
        self.assertEqual(ir.segments[0].n_samples, 110)
        self.assertAlmostEqual(
            float(ir.segments[0].parts[0].planes["H"].start.freqs_hz[0]), f0, places=12
        )

        q_ir, q_info = quantize_program_ir(
            ir,
            plane_to_channel={"H": 0, "V": 1, "A": 2, "B": 3},
        )
        self.assertEqual(q_ir.segments[0].n_samples, 96)
        self.assertEqual(q_info[0].original_samples, 110)
        self.assertEqual(q_info[0].quantized_samples, 96)

        seg_len_s = 96 / fs
        expected = round(f0 * seg_len_s) / seg_len_s
        snapped = q_ir.segments[0].parts[0].planes["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(snapped), expected, places=12)

    def test_loop_n_hold_does_not_snap(self) -> None:
        fs = 10.0
        f0 = 7.0
        b = (
            AWGProgramBuilder(sample_rate=fs)
            .plane("H")
            .define("dH", plane="H", freqs=[f0], amps=[1.0], phases="auto")
            .segment("hold", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=0.25, warn_df=0.1)
        )
        ir = b.build_ir()
        f = ir.segments[0].parts[0].planes["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(f), f0, places=12)

    def test_segment_with_only_time_zero_ops_errors(self) -> None:
        b = AWGProgramBuilder(sample_rate=10.0).plane("H")
        b.define("dH", plane="H", freqs=[1.0], amps=[1.0], phases="auto")
        b.segment("bad", mode="once")
        b.tones("H").use_def("dH").move(df=1.0, time=0.0)
        with self.assertRaises(ValueError):
            b.build_ir()
