import unittest

import numpy as np

from awgsegmentfactory.intent_ir import InterpSpec
from awgsegmentfactory.interpolation import interp_param
from awgsegmentfactory.quantize import quantize_resolved_ir
from awgsegmentfactory.resolved_ir import (
    ResolvedIR,
    ResolvedLogicalChannelPart,
    ResolvedPart,
    ResolvedSegment,
)
from awgsegmentfactory.resolved_timeline import LogicalChannelState
from awgsegmentfactory.synth_samples import compile_sequence_program


class TestInterpolation(unittest.TestCase):
    def test_adiabatic_ramp_shapes_match_sample_synth(self) -> None:
        # Sample synthesis passes u as (n_samples, 1) so it broadcasts over tones.
        n_samples = 10
        n_tones = 8
        u = (np.arange(n_samples, dtype=float) / float(n_samples))[:, None]

        start = np.full((n_tones,), 0.7, dtype=float)
        end = np.full((n_tones,), 0.0, dtype=float)

        out = interp_param(start, end, interp=InterpSpec("adiabatic_ramp"), u=u, t_s=None)
        self.assertEqual(out.shape, (n_samples, n_tones))

    def test_adiabatic_ramp_multi_tone_compiles(self) -> None:
        fs = 1000.0
        n = 96  # already satisfies step=32 and min size for 4 channels

        st0 = LogicalChannelState(
            freqs_hz=np.array([10.0, 20.0], dtype=float),
            amps=np.array([1.0, 1.0], dtype=float),
            phases_rad=np.array([0.0, 0.0], dtype=float),
        )
        st1 = LogicalChannelState(
            freqs_hz=st0.freqs_hz.copy(),
            amps=np.array([0.0, 0.0], dtype=float),
            phases_rad=st0.phases_rad.copy(),
        )
        empty = LogicalChannelState(
            freqs_hz=np.zeros((0,), dtype=float),
            amps=np.zeros((0,), dtype=float),
            phases_rad=np.zeros((0,), dtype=float),
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
                            start=st0, end=st1, interp=InterpSpec("adiabatic_ramp")
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

        quantized = quantize_resolved_ir(ir)
        prog = compile_sequence_program(
            quantized,
            gain=1.0,
            clip=1.0,
            full_scale=20000,
        )
        self.assertEqual(prog.segments[0].data_i16.shape, (4, n))
