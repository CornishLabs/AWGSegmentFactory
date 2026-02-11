import unittest

import numpy as np

from awgsegmentfactory.calibration import AODSin2Calib, AWGPhysicalSetupInfo
from awgsegmentfactory.intent_ir import InterpSpec
from awgsegmentfactory.resolved_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from awgsegmentfactory.synth_samples import QIRtoSamplesSegmentCompiler
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.resolved_timeline import LogicalChannelState


def _empty_logical_channel_state() -> LogicalChannelState:
    return LogicalChannelState(
        freqs_hz=np.zeros((0,), dtype=float),
        amps=np.zeros((0,), dtype=float),
        phases_rad=np.zeros((0,), dtype=float),
    )


def _identity_setup(logical_channels: tuple[str, ...]) -> AWGPhysicalSetupInfo:
    return AWGPhysicalSetupInfo.identity(logical_channels)


def _compile_to_card(
    q,
    *,
    physical_setup: AWGPhysicalSetupInfo,
    full_scale_mv: float,
    clip: float,
    full_scale: int,
    gpu: bool = False,
    output: str = "numpy",
) -> QIRtoSamplesSegmentCompiler:
    return QIRtoSamplesSegmentCompiler(
        quantised=q,
        physical_setup=physical_setup,
        full_scale_mv=full_scale_mv,
        clip=clip,
        full_scale=full_scale,
    ).compile_to_card_int16(gpu=gpu, output=output)


def _unit_sin2_calib() -> AODSin2Calib:
    return AODSin2Calib(
        g_poly_high_to_low=(1.0,),
        v0_a_poly_high_to_low=(1.0,),
        freq_min_hz=0.0,
        freq_max_hz=1.0,
        min_v0_sq=1e-30,
    )


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
            _compile_to_card(
                q,
                physical_setup=_identity_setup(q.logical_channels),
                full_scale_mv=1.0,
                clip=1.0,
                full_scale=20000,
                output="cupy",
            )
        with self.assertRaises(ValueError):
            QIRtoSamplesSegmentCompiler(
                quantised=q,
                physical_setup=_identity_setup(q.logical_channels),
                full_scale_mv=1.0,
                clip=1.0,
                full_scale=20000,
            ).compile_to_voltage_mV(output="cupy")

    def test_compile_to_voltage_mv_does_not_populate_int16_slots(self) -> None:
        fs = 1000.0
        n = 96
        st = LogicalChannelState(
            freqs_hz=np.array([10.0], dtype=float),
            amps=np.array([1.0], dtype=float),
            phases_rad=np.array([0.0], dtype=float),
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
            phase_mode="manual",
        )
        ir = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H", "V", "A", "B"),
            segments=(seg0,),
        )
        q = quantize_resolved_ir(ir)
        repo = QIRtoSamplesSegmentCompiler(
            quantised=q,
            physical_setup=_identity_setup(q.logical_channels),
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=20000,
        )
        voltage = repo.compile_to_voltage_mV()
        self.assertEqual(len(voltage), 1)
        self.assertEqual(voltage[0].segment_index, 0)
        self.assertEqual(voltage[0].name, "s0")
        self.assertEqual(voltage[0].data_mV.shape, (4, n))
        self.assertEqual(repo.compiled_indices, ())
        with self.assertRaises(ValueError):
            _ = repo.segments

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
        prog_continue = _compile_to_card(
            q_continue,
            physical_setup=_identity_setup(q_continue.logical_channels),
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_manual = _compile_to_card(
            q_manual,
            physical_setup=_identity_setup(q_manual.logical_channels),
            full_scale_mv=1.0,
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
        prog_continue = _compile_to_card(
            q_continue,
            physical_setup=_identity_setup(q_continue.logical_channels),
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_manual = _compile_to_card(
            q_manual,
            physical_setup=_identity_setup(q_manual.logical_channels),
            full_scale_mv=1.0,
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
        prog = _compile_to_card(
            q,
            physical_setup=_identity_setup(q.logical_channels),
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=20000,
        )
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
        setup_uncal = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(None,),
        )
        calib = _unit_sin2_calib()
        setup_cal = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(calib,),
        )
        prog_uncal = _compile_to_card(
            q,
            physical_setup=setup_uncal,
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        prog_cal = _compile_to_card(
            q,
            physical_setup=setup_cal,
            full_scale_mv=1.0,
            clip=1.0,
            full_scale=full_scale,
        )
        self.assertEqual(int(prog_uncal.segments[0].data_i16[0, 0]), 2000)
        expected_rf = float(
            np.asarray(
                calib.rf_amps(
                    np.array([10.0], dtype=float),
                    np.array([0.1], dtype=float),
                    logical_channel="H",
                    xp=np,
                ),
                dtype=float,
            )[0]
        )
        self.assertEqual(
            int(prog_cal.segments[0].data_i16[0, 0]),
            int(np.rint(expected_rf * float(full_scale))),
        )

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
            calib = _unit_sin2_calib()
            setup_cal = AWGPhysicalSetupInfo(
                logical_to_hardware_map={"H": 0},
                channel_calibrations=(calib,),
            )
            _compile_to_card(
                q,
                physical_setup=setup_cal,
                full_scale_mv=1.0,
                clip=1.0,
                full_scale=20000,
            )

        expected_amps = np.asarray(
            calib.rf_amps(st.freqs_hz, st.amps, logical_channel="H", xp=np),
            dtype=float,
        ).reshape(-1)
        np.testing.assert_allclose(captured["amps"], expected_amps)
