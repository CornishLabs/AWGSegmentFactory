import unittest

import numpy as np

from awgsegmentfactory.calibration import AODSin2Calib
from awgsegmentfactory.intent_ir import InterpSpec
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.resolved_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from awgsegmentfactory.resolved_timeline import LogicalChannelState
from awgsegmentfactory.synth_samples import compile_sequence_program


class TestAODSin2Calib(unittest.TestCase):
    def test_inverse_matches_forward_for_constant_coeffs(self) -> None:
        # g(freq)=1, v0(freq)=1 (approximately), so:
        #   y = sin^2((pi/2)*(a/v0))  ->  a = v0*(2/pi)*arcsin(sqrt(y))
        calib = AODSin2Calib(
            g_poly_by_logical_channel={"*": (1.0,)},
            v0_a_poly_by_logical_channel={"*": (1.0,)},
            freq_mid_hz=0.0,
            freq_halfspan_hz=1.0,
            amp_scale=1.0,
            min_g=1e-12,
            min_v0_sq=1e-30,
            y_eps=1e-6,
        )

        y = np.linspace(0.0, 0.95, 32, dtype=float)
        a = np.asarray(calib.rf_amps(np.zeros_like(y), y, logical_channel="H", xp=np), dtype=float)
        v0 = np.sqrt(1.0 + float(calib.min_v0_sq))
        y_hat = np.sin((0.5 * np.pi) * (a / v0)) ** 2
        np.testing.assert_allclose(y_hat, y, rtol=0.0, atol=2e-12)

    def test_used_in_sample_synthesis_pipeline(self) -> None:
        fs = 1000.0
        n = 384  # min size for 1 channel with default step=32
        full_scale = 20000

        st = LogicalChannelState(
            freqs_hz=np.array([10.0], dtype=float),
            amps=np.array([0.25], dtype=float),  # interpreted as "optical power"
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

        calib = AODSin2Calib(
            g_poly_by_logical_channel={"*": (1.0,)},
            v0_a_poly_by_logical_channel={"*": (1.0,)},
            freq_mid_hz=0.0,
            freq_halfspan_hz=1.0,
            amp_scale=1.0,
            min_v0_sq=1e-30,
        )

        ir = ResolvedIR(
            sample_rate_hz=fs,
            logical_channels=("H",),
            segments=(seg0,),
        )
        q = quantize_resolved_ir(ir, logical_channel_to_hardware_channel={"H": 0})
        prog = compile_sequence_program(
            q,
            gain=1.0,
            clip=1.0,
            full_scale=full_scale,
            optical_power_calib=calib,
        )

        expected_amp = float((2.0 / np.pi) * np.arcsin(np.sqrt(0.25)))
        expected_i16 = int(np.rint(expected_amp * float(full_scale)))
        self.assertEqual(int(prog.segments[0].data_i16[0, 0]), expected_i16)
