import unittest

import numpy as np

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.calibration import AWGPhysicalSetupInfo
from awgsegmentfactory.intent_ir import HoldOp, RemapFromDefOp, UseDefOp
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.synth_samples import compile_sequence_program


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
            .hold(time=hold_s)
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        self.assertEqual(ir.segments[0].n_samples, 110)
        self.assertAlmostEqual(
            float(ir.segments[0].parts[0].logical_channels["H"].start.freqs_hz[0]),
            f0,
            places=12,
        )

        quantized = quantize_resolved_ir(ir)
        q_ir = quantized.resolved_ir
        q_info = quantized.quantization
        self.assertEqual(q_ir.segments[0].n_samples, 96)
        self.assertEqual(q_info[0].original_samples, 110)
        self.assertEqual(q_info[0].quantized_samples, 96)

        seg_len_s = 96 / fs
        expected = round(f0 * seg_len_s) / seg_len_s
        snapped = q_ir.segments[0].parts[0].logical_channels["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(snapped), expected, places=12)

    def test_wait_trig_can_skip_quantum_snapping_for_min_latency(self) -> None:
        fs = 10_000_000.0
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .logical_channel("V")
            .segment("sync", mode="wait_trig", snap_len_to_quantum=False)
            .hold(time=1.0 / fs)  # exactly 1 sample
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        self.assertEqual(ir.segments[0].n_samples, 1)

        quantized = quantize_resolved_ir(ir)
        q_ir = quantized.resolved_ir
        q_info = quantized.quantization

        self.assertFalse(q_info[0].snap_len_to_quantum)
        self.assertEqual(q_ir.segments[0].n_samples, 192)  # min for 2 channels (step=32)
        self.assertGreater(q_info[0].quantum_samples, q_ir.segments[0].n_samples)

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
            .hold(time=0.25)
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

    def test_add_and_remove_tones(self) -> None:
        fs = 1000.0
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .logical_channel("V")
            .logical_channel("A")
            .logical_channel("B")
            .define("v0", logical_channel="V", freqs=[100.0], amps=[0.0], phases="auto")
            .segment("s0", mode="once")
            .tones("V")
            .use_def("v0")
            .hold(time=0.1)
            .segment("s_add", mode="once", phase_mode="manual")
            .tones("V")
            .add_tone(f=200.0)
            .ramp_amp_to(amps=1.0, idxs=[1], time=0.1)
            .segment("s_remove", mode="once", phase_mode="manual")
            .tones("V")
            .remove_tones(idxs=[1])
            .hold(time=0.1)
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        seg_add = next(s for s in ir.segments if s.name == "s_add")
        self.assertEqual(seg_add.phase_mode, "manual")
        self.assertEqual(len(seg_add.parts), 1)
        v_part = seg_add.parts[0].logical_channels["V"]
        self.assertEqual(v_part.start.freqs_hz.shape, (2,))
        np.testing.assert_allclose(v_part.start.freqs_hz, [100.0, 200.0])
        self.assertEqual(v_part.end.amps.shape, (2,))
        self.assertAlmostEqual(float(v_part.end.amps[1]), 1.0, places=12)

        quantized = quantize_resolved_ir(ir)
        prog = compile_sequence_program(
            quantized,
            physical_setup=AWGPhysicalSetupInfo.identity(quantized.logical_channels),
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=20000,
        )
        self.assertGreater(len(prog.segments), 0)

    def test_parallel_ops_share_one_time_interval(self) -> None:
        fs = 1000.0
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .logical_channel("V")
            .define("h0", logical_channel="H", freqs=[10.0], amps=[0.5], phases="auto")
            .define("v0", logical_channel="V", freqs=[20.0], amps=[0.5], phases="auto")
            .segment("init", mode="once")
            .tones("H")
            .use_def("h0")
            .tones("V")
            .use_def("v0")
            .hold(time=0.1)
            .segment("ramp_off", mode="once")
            .parallel(
                lambda p: (
                    p.tones("H").ramp_amp_to(amps=0.0, time=0.2),
                    p.tones("V").ramp_amp_to(amps=0.0, time=0.2),
                )
            )
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        seg = next(s for s in ir.segments if s.name == "ramp_off")
        self.assertEqual(seg.n_samples, 200)
        self.assertEqual(len(seg.parts), 1)
        part = seg.parts[0]
        self.assertEqual(part.logical_channels["H"].interp.kind, "linear")
        self.assertEqual(part.logical_channels["V"].interp.kind, "linear")
