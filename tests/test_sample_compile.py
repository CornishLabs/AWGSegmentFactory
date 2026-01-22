import unittest

import numpy as np

from awgsegmentfactory.program_ir import PartIR, PlanePartIR, ProgramIR, SegmentIR
from awgsegmentfactory.sample_compile import compile_sequence_program
from awgsegmentfactory.timeline import PlaneState


def _empty_plane_state() -> PlaneState:
    return PlaneState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


class TestSampleCompile(unittest.TestCase):
    def test_phase_mode_carry_vs_fixed(self) -> None:
        fs = 1000.0
        f = 10.0
        n = 96  # already satisfies step=32 and min size for 4 channels
        phase0 = 0.0

        st = PlaneState(
            freqs_hz=np.array([f], dtype=float),
            amps=np.array([1.0], dtype=float),
            phases_rad=np.array([phase0], dtype=float),
        )
        empty = _empty_plane_state()

        seg0 = SegmentIR(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                PartIR(
                    n_samples=n,
                    planes={
                        "H": PlanePartIR(start=st, end=st, interp="hold"),
                        "V": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "A": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "B": PlanePartIR(start=empty, end=empty, interp="hold"),
                    },
                ),
            ),
            phase_mode="carry",
        )

        seg1_carry = SegmentIR(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=seg0.parts,
            phase_mode="carry",
        )
        seg1_fixed = SegmentIR(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=seg0.parts,
            phase_mode="fixed",
        )

        ir_carry = ProgramIR(
            sample_rate_hz=fs, planes=("H", "V", "A", "B"), segments=(seg0, seg1_carry)
        )
        ir_fixed = ProgramIR(
            sample_rate_hz=fs, planes=("H", "V", "A", "B"), segments=(seg0, seg1_fixed)
        )

        full_scale = 20000
        prog_carry = compile_sequence_program(
            ir_carry,
            plane_to_channel={"H": 0, "V": 1, "A": 2, "B": 3},
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_fixed = compile_sequence_program(
            ir_fixed,
            plane_to_channel={"H": 0, "V": 1, "A": 2, "B": 3},
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )

        # For fixed mode, segment 1 starts at phase0 -> sin(0)=0.
        self.assertEqual(int(prog_fixed.segments[1].data_i16[0, 0]), 0)

        # For carry mode, segment 1 starts at the end phase of segment 0.
        dphi = 2.0 * np.pi * f / fs
        phase_end = (phase0 + n * dphi) % (2.0 * np.pi)
        expected = int(np.round(np.sin(phase_end) * full_scale))
        self.assertEqual(int(prog_carry.segments[1].data_i16[0, 0]), expected)

    def test_carry_with_tone_count_change_errors(self) -> None:
        fs = 1000.0
        n = 96
        empty = _empty_plane_state()

        st2 = PlaneState(
            freqs_hz=np.array([10.0, 20.0]),
            amps=np.array([1.0, 1.0]),
            phases_rad=np.array([0.0, 0.0]),
        )
        st1 = PlaneState(
            freqs_hz=np.array([10.0]),
            amps=np.array([1.0]),
            phases_rad=np.array([0.0]),
        )

        seg0 = SegmentIR(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                PartIR(
                    n_samples=n,
                    planes={
                        "H": PlanePartIR(start=st2, end=st2, interp="hold"),
                        "V": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "A": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "B": PlanePartIR(start=empty, end=empty, interp="hold"),
                    },
                ),
            ),
            phase_mode="carry",
        )
        seg1 = SegmentIR(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=(
                PartIR(
                    n_samples=n,
                    planes={
                        "H": PlanePartIR(start=st1, end=st1, interp="hold"),
                        "V": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "A": PlanePartIR(start=empty, end=empty, interp="hold"),
                        "B": PlanePartIR(start=empty, end=empty, interp="hold"),
                    },
                ),
            ),
            phase_mode="carry",
        )

        ir = ProgramIR(
            sample_rate_hz=fs, planes=("H", "V", "A", "B"), segments=(seg0, seg1)
        )

        with self.assertRaises(ValueError):
            compile_sequence_program(
                ir,
                plane_to_channel={"H": 0, "V": 1, "A": 2, "B": 3},
                gain=1.0,
                clip=1.0,
                full_scale=20000,
            )
