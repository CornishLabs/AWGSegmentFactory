import unittest

import numpy as np

from awgsegmentfactory import AWGProgramBuilder, compile_sequence_program, quantize_resolved_ir
from awgsegmentfactory.calibration import AODSin2Calib, AWGPhysicalSetupInfo
from awgsegmentfactory.quantize import QuantizedIR


def _assert_encodable(obj) -> None:
    allowed = (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        tuple,
        list,
        dict,
        set,
        np.ndarray,
        np.generic,
    )
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_encodable(k)
            _assert_encodable(v)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _assert_encodable(v)
        return
    if not isinstance(obj, allowed):
        raise AssertionError(f"Found non-encodable type: {type(obj).__name__}")


class TestQuantizedIRCodec(unittest.TestCase):
    def test_encode_decode_roundtrip_preserves_samples(self) -> None:
        fs = 4.0  # small so we get a tiny IR quickly; quantize will enforce min segment size.
        calib = AODSin2Calib(
            g_poly_high_to_low=(1.0,),
            v0_a_poly_high_to_low=(1.0,),
            freq_min_hz=0.0,
            freq_max_hz=1.0,
            amp_scale=1.0,
        )

        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .define("dH", logical_channel="H", freqs=[7.0], amps=[0.25], phases="auto")
            .segment("s0", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=0.25)
        )

        ir = b.build_resolved_ir(sample_rate_hz=fs)
        q = quantize_resolved_ir(ir)

        encoded = q.encode()
        _assert_encodable(encoded)
        self.assertNotIn("calibrations", encoded.get("resolved_ir", {}))

        q2 = QuantizedIR.decode(encoded)
        physical_setup = AWGPhysicalSetupInfo(
            logical_to_hardware_map={"H": 0},
            channel_calibrations=(calib,),
        )

        c0 = compile_sequence_program(
            q,
            physical_setup=physical_setup,
            gain=1.0,
            clip=0.9,
            full_scale=32767,
        )
        c1 = compile_sequence_program(
            q2,
            physical_setup=physical_setup,
            gain=1.0,
            clip=0.9,
            full_scale=32767,
        )

        self.assertEqual(c0.steps, c1.steps)
        self.assertEqual(c0.quantization, c1.quantization)
        self.assertEqual(len(c0.segments), len(c1.segments))
        for s0, s1 in zip(c0.segments, c1.segments, strict=True):
            self.assertEqual(s0.name, s1.name)
            self.assertEqual(s0.n_samples, s1.n_samples)
            np.testing.assert_array_equal(s0.data_i16, s1.data_i16)
