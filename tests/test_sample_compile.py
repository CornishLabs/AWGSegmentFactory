import unittest

import numpy as np

from awgsegmentfactory.calibration import OpticalPowerToRFAmpCalib
from awgsegmentfactory.intent_ir import InterpSpec
from awgsegmentfactory.resolved_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from awgsegmentfactory.synth_samples import compile_sequence_program
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.resolved_timeline import LogicalChannelState


def _empty_logical_channel_state() -> LogicalChannelState:
    return LogicalChannelState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


class _ScaleOpticalPowerToRFAmpCalib(OpticalPowerToRFAmpCalib):
    def __init__(self, scale: float):
        self._scale = float(scale)

    def rf_amps(
        self,
        freqs_hz,
        optical_powers,
        *,
        logical_channel: str,
        xp=np,
    ):
        _ = (freqs_hz, logical_channel)
        return xp.asarray(optical_powers, dtype=float) * float(self._scale)


class TestSampleCompile(unittest.TestCase):
    def test_output_cupy_requires_gpu(self) -> None:
        fs = 1000.0
        n = 96
        empty = _empty_logical_channel_state()
        st = LogicalChannelState(
            freqs_hz=np.array([10.0], dtype=float),
            amps=np.array([1.0], dtype=float),
            phases_rad=np.array([0.0], dtype=float),
        )
        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st, end=st, interp=InterpSpec("hold")
                        ),
                        "V": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "A": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "B": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="continue",
        )
        ir = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0,),
        )
        q = quantize_resolved_ir(ir)
        with self.assertRaises(ValueError):
            compile_sequence_program(q, gain=1.0, clip=1.0, full_scale=20000, output="cupy")

    def test_phase_mode_continue_vs_manual(self) -> None:
        fs = 1000.0
        f = 10.0
        n = 96  # already satisfies step=32 and min size for 4 channels
        phase0 = 0.0

        st = LogicalChannelState(
            freqs_hz=np.array([f], dtype=float),
            amps=np.array([1.0], dtype=float),
            phases_rad=np.array([phase0], dtype=float),
        )
        empty = _empty_logical_channel_state()

        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st, end=st, interp=InterpSpec("hold")
                        ),
                        "V": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "A": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "B": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="continue",
        )

        seg1_continue = ResolvedSegment(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=seg0.parts,
            phase_mode="continue",
        )
        seg1_manual = ResolvedSegment(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=seg0.parts,
            phase_mode="manual",
        )

        ir_continue = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0, seg1_continue),
        )
        ir_manual = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0, seg1_manual),
        )

        full_scale = 20000
        q_continue = quantize_resolved_ir(ir_continue)
        q_manual = quantize_resolved_ir(ir_manual)
        prog_continue = compile_sequence_program(
            q_continue,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_manual = compile_sequence_program(
            q_manual,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )

        # For manual mode, segment 1 uses its declared start phase (phase0 -> sin(0)=0).
        self.assertEqual(int(prog_manual.segments[1].data_i16[0, 0]), 0)

        # For continue mode, segment 1 starts at the end phase of segment 0.
        dphi = 2.0 * np.pi * f / fs
        phase_end = (phase0 + n * dphi) % (2.0 * np.pi)
        expected = int(np.round(np.sin(phase_end) * full_scale))
        self.assertEqual(int(prog_continue.segments[1].data_i16[0, 0]), expected)

    def test_continue_with_tone_count_change_matches_by_frequency(self) -> None:
        fs = 1000.0
        f0 = 10.0
        n = 96
        empty = _empty_logical_channel_state()

        st2 = LogicalChannelState(
            freqs_hz=np.array([f0, 20.0]),
            amps=np.array([1.0, 1.0]),
            phases_rad=np.array([0.0, 0.0]),
        )
        st1 = LogicalChannelState(
            freqs_hz=np.array([f0]),
            amps=np.array([1.0]),
            phases_rad=np.array([1.234]),
        )

        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st2, end=st2, interp=InterpSpec("hold")
                        ),
                        "V": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "A": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "B": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="continue",
        )
        seg1_continue = ResolvedSegment(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st1, end=st1, interp=InterpSpec("hold")
                        ),
                        "V": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "A": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "B": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="continue",
        )
        seg1_manual = ResolvedSegment(
            name="s1",
            mode="loop_n",
            loop=1,
            parts=seg1_continue.parts,
            phase_mode="manual",
        )

        ir_continue = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0, seg1_continue),
        )
        ir_manual = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0, seg1_manual),
        )

        full_scale = 20000
        q_continue = quantize_resolved_ir(ir_continue)
        q_manual = quantize_resolved_ir(ir_manual)
        prog_continue = compile_sequence_program(
            q_continue,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_manual = compile_sequence_program(
            q_manual,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
        )

        # In continue mode with tone-count mismatch, matching is by frequency (tone f0 carries).
        dphi = 2.0 * np.pi * f0 / fs
        phase_end0 = (0.0 + n * dphi) % (2.0 * np.pi)
        expected_carry = int(np.round(np.sin(phase_end0) * full_scale))
        self.assertEqual(int(prog_continue.segments[1].data_i16[0, 0]), expected_carry)

        # In manual mode, segment 1 uses its own declared start phase.
        expected_fixed = int(np.round(np.sin(1.234) * full_scale))
        self.assertEqual(int(prog_manual.segments[1].data_i16[0, 0]), expected_fixed)

    def test_phase_mode_optimise_allows_empty_logical_channels(self) -> None:
        fs = 1000.0
        n = 96
        empty = _empty_logical_channel_state()
        st = LogicalChannelState(
            freqs_hz=np.array([10.0], dtype=float),
            amps=np.array([1.0], dtype=float),
            phases_rad=np.array([0.0], dtype=float),
        )
        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st, end=st, interp=InterpSpec("hold")
                        ),
                        "V": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "A": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                        "B": ResolvedLogicalChannelPart(
                            start=empty, end=empty, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="optimise",
        )
        ir = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0,),
        )
        q = quantize_resolved_ir(ir)
        prog = compile_sequence_program(q, gain=1.0, clip=1.0, full_scale=20000)
        self.assertEqual(prog.segments[0].data_i16.shape, (4, n))

    def test_optical_power_calib_scales_synth_amplitudes(self) -> None:
        fs = 1000.0
        n = 384  # min size for 1 channel with default step=32
        st = LogicalChannelState(
            freqs_hz=np.array([10.0], dtype=float),
            amps=np.array([0.1], dtype=float),
            phases_rad=np.array([np.pi / 2.0], dtype=float),
        )
        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st, end=st, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="manual",
        )
        ir = ResolvedIR(sample_rate_hz=fs, logical_channels=("H",), segments=(seg0,))
        q = quantize_resolved_ir(ir)
        full_scale = 20000
        prog_uncal = compile_sequence_program(
            q, gain=1.0, clip=1.0, full_scale=full_scale
        )
        prog_cal = compile_sequence_program(
            q,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
            optical_power_calib=_ScaleOpticalPowerToRFAmpCalib(2.0),
        )
        self.assertEqual(int(prog_uncal.segments[0].data_i16[0, 0]), 2000)
        self.assertEqual(int(prog_cal.segments[0].data_i16[0, 0]), 4000)

    def test_optical_power_calib_used_for_phase_optimisation(self) -> None:
        from unittest.mock import patch

        fs = 1000.0
        n = 384
        st = LogicalChannelState(
            freqs_hz=np.array([10.0, 20.0], dtype=float),
            amps=np.array([0.1, 0.2], dtype=float),
            phases_rad=np.array([0.0, 0.0], dtype=float),
        )
        seg0 = ResolvedSegment(
            name="s0",
            mode="loop_n",
            loop=1,
            parts=(
                ResolvedPart(
                    n_samples=n,
                    logical_channels={
                        "H": ResolvedLogicalChannelPart(
                            start=st, end=st, interp=InterpSpec("hold")
                        ),
                    },
                ),
            ),
            phase_mode="optimise",
        )
        ir = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H",),
            segments=(seg0,),
        )
        q = quantize_resolved_ir(ir)

        captured: dict[str, np.ndarray] = {}

        def fake_opt(*, freqs_hz, amps, phases_init_rad=None, fixed_mask=None):
            _ = (freqs_hz, phases_init_rad, fixed_mask)
            captured["amps"] = np.asarray(amps, dtype=float).reshape(-1)
            return np.zeros((len(captured["amps"]),), dtype=float)

        with patch(
            "awgsegmentfactory.synth_samples._optimise_phases_for_crest",
            side_effect=fake_opt,
        ):
            compile_sequence_program(
                q,
                gain=1.0,
                clip=1.0,
                full_scale=20000,
                optical_power_calib=_ScaleOpticalPowerToRFAmpCalib(3.0),
            )

        np.testing.assert_allclose(captured["amps"], np.array([0.3, 0.6], dtype=float))
