import unittest
from unittest.mock import patch

import numpy as np

from awgsegmentfactory.upload import (
    CPUUploadSession,
    SCAPPUploadSession,
    upload_sequence_program,
)


class _FakeCompiledSegment:
    def __init__(self, *, name: str, n_samples: int, data_i16):
        self.name = name
        self.n_samples = int(n_samples)
        self.data_i16 = data_i16


class _FakeStep:
    def __init__(
        self,
        *,
        step_index: int,
        segment_index: int,
        next_step: int,
        loops: int,
        on_trig: bool,
    ):
        self.step_index = int(step_index)
        self.segment_index = int(segment_index)
        self.next_step = int(next_step)
        self.loops = int(loops)
        self.on_trig = bool(on_trig)


class _FakeQuantisedSegment:
    def __init__(self, n_samples: int):
        self.n_samples = int(n_samples)


class _FakeQuantised:
    def __init__(self, segment_lengths: tuple[int, ...]):
        self.segments = tuple(_FakeQuantisedSegment(n) for n in segment_lengths)


class _FakePhysicalSetup:
    def __init__(self, n_ch: int):
        self.N_ch = int(n_ch)


class _FakeRepo:
    def __init__(
        self,
        *,
        compiled_segments: tuple[_FakeCompiledSegment, ...],
        segment_lengths: tuple[int, ...],
        n_ch: int = 1,
    ):
        self._compiled_segments = compiled_segments
        self._compiled_indices = tuple(i for i, _s in enumerate(compiled_segments))
        self.quantised = _FakeQuantised(segment_lengths)
        self.physical_setup = _FakePhysicalSetup(n_ch=n_ch)
        self.steps = (_FakeStep(step_index=0, segment_index=0, next_step=0, loops=1, on_trig=False),)

    @property
    def compiled_indices(self):
        return self._compiled_indices

    @property
    def segments(self):
        return self._compiled_segments

    def compiled_segment_items(self):
        return tuple((i, seg) for i, seg in enumerate(self._compiled_segments))

    def compiled_segment(self, idx: int):
        return self._compiled_segments[int(idx)]

    def to_numpy(self):
        return self


class TestUploadAPI(unittest.TestCase):
    def test_auto_mode_routes_numpy_to_cpu(self) -> None:
        repo = _FakeRepo(
            compiled_segments=(
                _FakeCompiledSegment(
                    name="s0",
                    n_samples=32,
                    data_i16=np.zeros((1, 32), dtype=np.int16),
                ),
            ),
            segment_lengths=(32,),
            n_ch=1,
        )
        session = CPUUploadSession(
            card=object(),
            sequence=object(),
            segments_hw=(),
            steps_hw=(),
            n_channels=1,
            segment_lengths=(32,),
            steps_signature=((0, 0, 0, 1, False),),
        )
        with patch("awgsegmentfactory.upload._full_cpu_upload", return_value=session) as p:
            out = upload_sequence_program(repo, mode="auto", card=object(), upload_steps=True)
        self.assertIs(out, session)
        p.assert_called_once()

    def test_scapp_mode_rejects_numpy_buffers(self) -> None:
        repo = _FakeRepo(
            compiled_segments=(
                _FakeCompiledSegment(
                    name="s0",
                    n_samples=32,
                    data_i16=np.zeros((1, 32), dtype=np.int16),
                ),
            ),
            segment_lengths=(32,),
            n_ch=1,
        )
        with self.assertRaises(ValueError):
            upload_sequence_program(repo, mode="scapp", card=object(), upload_steps=True)

    def test_cpu_data_only_requires_cpu_session(self) -> None:
        repo = _FakeRepo(
            compiled_segments=(
                _FakeCompiledSegment(name="s0", n_samples=32, data_i16=np.zeros((1, 32), dtype=np.int16)),
            ),
            segment_lengths=(32,),
            n_ch=1,
        )
        wrong_session = SCAPPUploadSession(
            card=object(),
            n_channels=1,
            segment_lengths=(32,),
            steps_signature=((0, 0, 0, 1, False),),
        )
        with self.assertRaises(ValueError):
            upload_sequence_program(
                repo,
                mode="cpu",
                cpu_session=wrong_session,
                upload_steps=False,
            )

    def test_scapp_data_only_requires_scapp_session(self) -> None:
        repo = _FakeRepo(
            compiled_segments=(
                _FakeCompiledSegment(name="s0", n_samples=32, data_i16=object()),
            ),
            segment_lengths=(32,),
            n_ch=1,
        )
        wrong_session = CPUUploadSession(
            card=object(),
            sequence=object(),
            segments_hw=(),
            steps_hw=(),
            n_channels=1,
            segment_lengths=(32,),
            steps_signature=((0, 0, 0, 1, False),),
        )
        with self.assertRaises(ValueError):
            upload_sequence_program(
                repo,
                mode="scapp",
                cpu_session=wrong_session,
                upload_steps=False,
            )

    def test_scapp_data_only_routes_to_scapp_updater(self) -> None:
        repo = _FakeRepo(
            compiled_segments=(
                _FakeCompiledSegment(name="s0", n_samples=32, data_i16=object()),
            ),
            segment_lengths=(32,),
            n_ch=1,
        )
        session = SCAPPUploadSession(
            card=object(),
            n_channels=1,
            segment_lengths=(32,),
            steps_signature=((0, 0, 0, 1, False),),
        )
        with patch(
            "awgsegmentfactory.upload._update_scapp_segments_only",
            return_value=session,
        ) as p:
            out = upload_sequence_program(
                repo,
                mode="scapp",
                cpu_session=session,
                upload_steps=False,
                segment_indices=[0],
            )
        self.assertIs(out, session)
        p.assert_called_once()


if __name__ == "__main__":
    unittest.main()

