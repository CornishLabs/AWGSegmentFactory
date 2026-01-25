import unittest

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.intent_ir import HoldOp, RemapFromDefOp, UseDefOp
from awgsegmentfactory.quantize import quantize_resolved_ir


class TestBuilder(unittest.TestCase):
    def test_build_ir_basic_hold_is_quantized(self) -> None:
        fs = 4.0  # dt=0.25s
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .define("dH", logical_channel="H", freqs=[100.0], amps=[1.0], phases="auto")
            .segment("seg0", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=0.3)
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        self.assertEqual(len(ir.segments), 1)
        self.assertEqual(ir.segments[0].name, "seg0")
        self.assertEqual(ir.segments[0].mode, "loop_n")  # "once" becomes loop_n, loop=1
        self.assertEqual(ir.segments[0].loop, 1)
        self.assertEqual(ir.segments[0].n_samples, 2)  # ceil(0.3 * 4) = 2

        part = ir.segments[0].parts[0]
        self.assertEqual(part.n_samples, 2)
        np.testing.assert_allclose(part.logical_channels["H"].start.freqs_hz, [100.0])
        np.testing.assert_allclose(part.logical_channels["H"].start.amps, [1.0])

        tl = b.build_timeline(sample_rate_hz=fs)
        self.assertEqual(tl.t_end, ir.duration_s)

    def test_build_spec_records_ops_and_expands_remap_dst_all(self) -> None:
        b = AWGProgramBuilder().logical_channel("H")
        b.define(
            "start",
            logical_channel="H",
            freqs=[1.0, 2.0, 3.0],
            amps=[1.0, 1.0, 1.0],
            phases="auto",
        )
        b.define(
            "target",
            logical_channel="H",
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

        spec = b.build_intent_ir()
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
            AWGProgramBuilder()
            .logical_channel("H")
            .logical_channel("V")
            .logical_channel("A")
            .logical_channel("B")
            .define("dH", logical_channel="H", freqs=[f0], amps=[1.0], phases="auto")
            .segment("wait", mode="wait_trig")
            .tones("H")
            .use_def("dH")
            .hold(time=hold_s, warn_df=0.1)
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        self.assertEqual(ir.segments[0].n_samples, 110)
        self.assertAlmostEqual(
            float(ir.segments[0].parts[0].logical_channels["H"].start.freqs_hz[0]),
            f0,
            places=12,
        )

        quantized = quantize_resolved_ir(
            ir,
            logical_channel_to_hardware_channel={"H": 0, "V": 1, "A": 2, "B": 3},
        )
        q_ir = quantized.resolved_ir
        q_info = quantized.quantization
        self.assertEqual(q_ir.segments[0].n_samples, 96)
        self.assertEqual(q_info[0].original_samples, 110)
        self.assertEqual(q_info[0].quantized_samples, 96)

        seg_len_s = 96 / fs
        expected = round(f0 * seg_len_s) / seg_len_s
        snapped = q_ir.segments[0].parts[0].logical_channels["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(snapped), expected, places=12)

    def test_loop_n_hold_does_not_snap(self) -> None:
        fs = 10.0
        f0 = 7.0
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .define("dH", logical_channel="H", freqs=[f0], amps=[1.0], phases="auto")
            .segment("hold", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=0.25, warn_df=0.1)
        )
        ir = b.build_resolved_ir(sample_rate_hz=fs)
        f = ir.segments[0].parts[0].logical_channels["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(f), f0, places=12)

    def test_segment_with_only_time_zero_ops_errors(self) -> None:
        b = AWGProgramBuilder().logical_channel("H")
        b.define("dH", logical_channel="H", freqs=[1.0], amps=[1.0], phases="auto")
        b.segment("bad", mode="once")
        b.tones("H").use_def("dH").move(df=1.0, time=0.0)
        with self.assertRaises(ValueError):
            b.build_resolved_ir(sample_rate_hz=10.0)
