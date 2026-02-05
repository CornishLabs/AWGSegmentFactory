import unittest

from awgsegmentfactory import AWGProgramBuilder, quantize_resolved_ir
from awgsegmentfactory.debug import (
    format_intent_ir,
    format_ir,
    format_quantized_ir,
    format_resolved_ir,
)


class TestDebugIRFormatting(unittest.TestCase):
    def test_formats_intent_and_resolved_ir(self) -> None:
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .define("dH", logical_channel="H", freqs=[100.0], amps=[1.0], phases="auto")
            .segment("s0", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=1e-6)
        )

        intent = b.build_intent_ir()
        s = format_intent_ir(intent)
        self.assertIn("IntentIR:", s)
        self.assertIn("'s0'", s)
        self.assertEqual(format_ir(b), s)

        ir = b.build_resolved_ir(sample_rate_hz=4.0)
        s2 = format_resolved_ir(ir)
        self.assertIn("ResolvedIR:", s2)
        self.assertIn("'s0'", s2)
        self.assertIn("part 0:", s2)
        self.assertIn("H:hold", s2)

        self.assertEqual(format_ir(intent), s)
        self.assertEqual(format_ir(ir), s2)

    def test_formats_quantized_ir(self) -> None:
        b = (
            AWGProgramBuilder()
            .logical_channel("H")
            .define("dH", logical_channel="H", freqs=[100.0], amps=[1.0], phases="auto")
            .segment("s0", mode="once")
            .tones("H")
            .use_def("dH")
            .hold(time=1e-6)
        )
        ir = b.build_resolved_ir(sample_rate_hz=4.0)
        q = quantize_resolved_ir(ir, logical_channel_to_hardware_channel={"H": 0})
        s = format_quantized_ir(q)
        self.assertIn("QuantizedIR:", s)
        self.assertIn("'s0'", s)
        self.assertIn("logical_channel_to_hardware_channel", s)
        self.assertEqual(format_ir(q), s)
