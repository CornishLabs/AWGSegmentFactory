from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ir import (
    ProgramSpec, SegmentSpec, SegmentMode,
    DefinitionSpec,
    HoldOp, UseDefOp, MoveOp, RampAmpToOp, RemapFromDefOp,
)
from .program_ir import ProgramIR
from .resolve import resolve_program, resolve_program_ir

def _as_tuple_f(x: float | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(x, (int, float)):
        return (float(x),)
    return tuple(float(v) for v in x)

def _phases_auto(n: int) -> Tuple[float, ...]:
    # placeholder: you can later implement phase picking for cresting, etc.
    return tuple(0.0 for _ in range(n))

class ToneView:
    """
    View over a particular plane (e.g. "H" or "V") within the current segment.
    Implements plane ops and forwards builder methods so fluent chaining works.
    """
    def __init__(self, b: "AWGProgramBuilder", plane: str):
        self._b = b
        self._plane = plane

    # ---- plane ops ----
    def use_def(self, def_name: str) -> "ToneView":
        self._b._append(UseDefOp(plane=self._plane, def_name=def_name))
        return self

    def move(self, *, df: float, time: float, idxs: Optional[Sequence[int]] = None, kind: str = "linear") -> "ToneView":
        idx_t = tuple(int(i) for i in idxs) if idxs is not None else None
        self._b._append(MoveOp(plane=self._plane, df_hz=float(df), time_s=float(time), idxs=idx_t, kind=kind))  # type: ignore[arg-type]
        return self

    def ramp_amp_to(
        self,
        *,
        amps: float | Sequence[float],
        time: float,
        kind: str = "linear",
        tau: Optional[float] = None,
        idxs: Optional[Sequence[int]] = None,
    ) -> "ToneView":
        idx_t = tuple(int(i) for i in idxs) if idxs is not None else None
        amps_t = amps if isinstance(amps, (int, float)) else tuple(float(a) for a in amps)
        self._b._append(
            RampAmpToOp(
                plane=self._plane,
                amps_target=amps_t,  # type: ignore[arg-type]
                time_s=float(time),
                kind=kind,  # type: ignore[arg-type]
                tau_s=float(tau) if tau is not None else None,
                idxs=idx_t,
            )
        )
        return self

    def remap_from_def(
        self,
        *,
        target_def: str,
        src: Sequence[int],
        dst: str | Sequence[int] = "all",
        time: float,
        kind: str = "min_jerk",
    ) -> "ToneView":
        src_t = tuple(int(i) for i in src)

        # dst="all" means 0..len(target_def)-1, but we don't know length here;
        # store "all" as empty sentinel and expand in build_spec() using definitions.
        # For simplicity: expand here by looking up definitions now.
        d = self._b._definitions[target_def]
        if dst == "all":
            dst_t = tuple(range(len(d.freqs_hz)))
        else:
            dst_t = tuple(int(i) for i in dst)

        self._b._append(
            RemapFromDefOp(
                plane=self._plane,
                target_def=target_def,
                src=src_t,
                dst=dst_t,
                time_s=float(time),
                kind=kind,  # type: ignore[arg-type]
            )
        )
        return self

    # ---- forward builder methods for fluent chaining ----
    def tones(self, plane: str) -> "ToneView":
        return self._b.tones(plane)

    def segment(self, name: str, mode: SegmentMode = "once", loop: Optional[int] = None) -> "AWGProgramBuilder":
        return self._b.segment(name, mode=mode, loop=loop)

    def hold(self, *, time: float, warn_df: Optional[float] = None) -> "AWGProgramBuilder":
        return self._b.hold(time=time, warn_df=warn_df)

    def build(self):
        return self._b.build()


class AWGProgramBuilder:
    def __init__(self, sample_rate: float):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self._fs = float(sample_rate)
        self._planes: List[str] = []
        self._definitions: Dict[str, DefinitionSpec] = {}
        self._segments: List[SegmentSpec] = []
        self._current_seg: Optional[int] = None
        self._calibrations: Dict[str, Any] = {}

    def with_calibration(self, key: str, obj: Any) -> "AWGProgramBuilder":
        self._calibrations[key] = obj
        return self

    def plane(self, name: str) -> "AWGProgramBuilder":
        if name in self._planes:
            return self
        self._planes.append(str(name))
        return self

    def define(
        self,
        name: str,
        *,
        plane: str,
        freqs: Sequence[float],
        amps: Sequence[float],
        phases: str | Sequence[float] = "auto",
    ) -> "AWGProgramBuilder":
        if plane not in self._planes:
            raise ValueError(f"Define references unknown plane {plane!r}. Call .plane({plane!r}) first.")
        if len(freqs) != len(amps):
            raise ValueError("define: freqs and amps must have same length")
        n = len(freqs)
        ph: Tuple[float, ...]
        if phases == "auto":
            ph = _phases_auto(n)
        else:
            if len(phases) != n:
                raise ValueError("define: phases length mismatch")
            ph = tuple(float(x) for x in phases)

        self._definitions[name] = DefinitionSpec(
            name=name,
            plane=plane,
            freqs_hz=tuple(float(x) for x in freqs),
            amps=tuple(float(x) for x in amps),
            phases_rad=ph,
        )
        return self

    def segment(self, name: str, mode: SegmentMode = "once", loop: Optional[int] = None) -> "AWGProgramBuilder":
        mode = str(mode)  # type: ignore[assignment]
        if mode == "once":
            mode = "loop_n"
            loop = 1
        if mode == "loop_n":
            if loop is None or loop <= 0:
                raise ValueError("loop_n requires loop=N with N>0")
        elif mode == "wait_trig":
            loop = 1
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        self._segments.append(SegmentSpec(name=name, mode=mode, loop=int(loop), ops=tuple()))
        self._current_seg = len(self._segments) - 1
        return self

    def tones(self, plane: str) -> ToneView:
        if plane not in self._planes:
            raise KeyError(f"Unknown plane {plane!r}")
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .tones(...)")
        return ToneView(self, plane)

    def hold(self, *, time: float, warn_df: Optional[float] = None) -> "AWGProgramBuilder":
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .hold(...)")
        self._append(HoldOp(time_s=float(time), warn_df_hz=float(warn_df) if warn_df is not None else None))
        return self

    def _append(self, op) -> None:
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before adding ops")
        seg = self._segments[self._current_seg]
        self._segments[self._current_seg] = SegmentSpec(
            name=seg.name, mode=seg.mode, loop=seg.loop, ops=seg.ops + (op,)
        )

    def build_spec(self) -> ProgramSpec:
        if not self._segments:
            raise RuntimeError("No segments defined")
        return ProgramSpec(
            sample_rate_hz=self._fs,
            planes=tuple(self._planes),
            definitions=dict(self._definitions),
            segments=tuple(self._segments),
            calibrations=dict(self._calibrations),
        )

    def build(self):
        spec = self.build_spec()
        return resolve_program(spec)

    def build_ir(self) -> ProgramIR:
        spec = self.build_spec()
        return resolve_program_ir(spec)
