import unittest

import numpy as np

from awgsegmentfactory.program_ir import PartIR, PlanePartIR, ProgramIR, SegmentIR
from awgsegmentfactory.sequence_compile import (
    format_samples_time,
    quantum_samples,
    quantize_program_ir,
)
from awgsegmentfactory.timeline import PlaneState


def _empty_plane_state() -> PlaneState:
    return PlaneState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


class TestSequenceCompile(unittest.TestCase):
    def test_quantum_samples_matches_625mhz_40us_example(self) -> None:
        q = quantum_samples(625e6, quantum_s=40e-6, step_samples=32)
        self.assertEqual(q, 24992)
        self.assertEqual(format_samples_time(q, 625e6), "24992 (39.9872 µs)")

    def test_quantize_non_loopable_segment_ceils_to_step(self) -> None:
        fs = 1000.0
        st = PlaneState(freqs_hz=np.array([10.0]), amps=np.array([1.0]), phases_rad=np.array([0.0]))
        seg = SegmentIR(
            name="seg",
            mode="loop_n",
            loop=1,
            parts=(
                PartIR(
                    n_samples=193,
                    planes={
                        "H": PlanePartIR(start=st, end=st, interp="hold"),
                        "V": PlanePartIR(start=_empty_plane_state(), end=_empty_plane_state(), interp="hold"),
                    },
                ),
            ),
        )
        ir = ProgramIR(sample_rate_hz=fs, planes=("H", "V"), segments=(seg,))

        q_ir, info = quantize_program_ir(ir, plane_to_channel={"H": 0, "V": 1})
        self.assertEqual(info[0].original_samples, 193)
        self.assertEqual(info[0].quantized_samples, 224)
        self.assertEqual(q_ir.segments[0].n_samples, 224)

        # Non-loopable segments do not get frequency wrap snapping.
        f = q_ir.segments[0].parts[0].planes["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(f), 10.0, places=12)

    def test_quantize_loopable_constant_segment_can_round_down_and_snaps_freqs(self) -> None:
        fs = 1000.0
        f0 = 7.0
        st = PlaneState(freqs_hz=np.array([f0]), amps=np.array([1.0]), phases_rad=np.array([0.0]))
        empty = _empty_plane_state()

        seg = SegmentIR(
            name="wait",
            mode="wait_trig",
            loop=1,
            parts=(
                PartIR(
                    n_samples=110,
                    planes={
                        "H": PlanePartIR(start=st, end=st, interp="hold"),
                        "V": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "A": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "B": PlanePartIR(start=empty, end=empty, interp="hold"),
                    },
                ),
            ),
        )
        ir = ProgramIR(sample_rate_hz=fs, planes=("H", "V", "A", "B"), segments=(seg,))

        q_ir, info = quantize_program_ir(ir, plane_to_channel={"H": 0, "V": 1, "A": 2, "B": 3})

        # 110 rounded to nearest multiple of 32 -> 96, and 4 active channels -> min segment is 96.
        self.assertEqual(info[0].original_samples, 110)
        self.assertEqual(info[0].quantized_samples, 96)
        self.assertEqual(q_ir.segments[0].n_samples, 96)

        seg_len_s = 96 / fs
        expected = round(f0 * seg_len_s) / seg_len_s
        f = q_ir.segments[0].parts[0].planes["H"].start.freqs_hz[0]
        self.assertAlmostEqual(float(f), expected, places=12)

