import json
import unittest
from dataclasses import replace

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.intent_ir import IntentDefinition, IntentIR


def _assert_basic_python(obj) -> None:
    allowed = (type(None), bool, int, float, str, dict, list, tuple)
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_basic_python(k)
            _assert_basic_python(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_basic_python(v)
        return
    if not isinstance(obj, allowed):
        raise AssertionError(f"Found non-basic type: {type(obj).__name__}")


class TestIntentIRCodec(unittest.TestCase):
    def test_encode_decode_roundtrip(self) -> None:
        b = AWGProgramBuilder().logical_channel("H").logical_channel("V")
        b.define("dH", logical_channel="H", freqs=[100.0], amps=[1.0], phases="auto")
        b.define("dV", logical_channel="V", freqs=[200.0], amps=[1.0], phases="auto")

        b.segment("init", mode="once", phase_mode="manual")
        b.tones("H").use_def("dH")
        b.tones("V").use_def("dV")
        b.hold(time=1e-6)

        b.segment("move_H", mode="once")
        b.parallel(
            lambda p: p.tones("H").move(df=+1.0, time=2e-6, idxs=[0]).tones("V").ramp_amp_to(
                amps=0.5, time=2e-6, idxs=[0]
            )
        )

        intent = b.build_intent_ir()
        encoded = intent.encode()
        _assert_basic_python(encoded)
        json.dumps(encoded, sort_keys=True)  # should be JSON-serializable

        decoded = IntentIR.decode(encoded)
        self.assertEqual(decoded, intent)

    def test_segment_fingerprints_change_with_referenced_definition(self) -> None:
        b = AWGProgramBuilder().logical_channel("H")
        b.define("d0", logical_channel="H", freqs=[10.0], amps=[0.1], phases="auto")
        b.define("d1", logical_channel="H", freqs=[10.0], amps=[0.2], phases="auto")

        b.segment("s0", mode="once")
        b.tones("H").use_def("d0")
        b.hold(time=1e-6)

        b.segment("s1", mode="once")
        b.tones("H").use_def("d1")
        b.hold(time=1e-6)

        intent = b.build_intent_ir()
        fp0 = intent.segment_fingerprints()

        defs2 = dict(intent.definitions)
        d0 = defs2["d0"]
        defs2["d0"] = IntentDefinition(
            name=d0.name,
            logical_channel=d0.logical_channel,
            freqs_hz=d0.freqs_hz,
            amps=(0.123,),
            phases_rad=d0.phases_rad,
        )
        intent2 = replace(intent, definitions=defs2)

        fp1 = intent2.segment_fingerprints()
        self.assertNotEqual(fp0[0]["digest"], fp1[0]["digest"])  # s0 references d0
        self.assertEqual(fp0[1]["digest"], fp1[1]["digest"])  # s1 references d1 only

        self.assertNotEqual(intent.digest(), intent2.digest())

