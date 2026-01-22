from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import (
    ProgramSpec,
    SegmentSpec,
    SegmentMode,
    SegmentPhaseMode,
    DefinitionSpec,
    HoldOp,
    UseDefOp,
    MoveOp,
    RampAmpToOp,
    RemapFromDefOp,
)
from .program_ir import ProgramIR
from .resolve import resolve_program, resolve_program_ir


def _phases_auto(n: int) -> Tuple[float, ...]:
    # placeholder: you can later implement phase picking for cresting, etc.
    return tuple(0.0 for _ in range(n))


class ToneView:
    """
    View over a particular logical channel (e.g. "H" or "V") within the current segment.
    Implements per-channel ops and forwards builder methods so fluent chaining works.
    """

    def __init__(self, b: "AWGProgramBuilder", logical_channel: str):
        self._b = b
        self._logical_channel = logical_channel

    # ---- per-logical-channel ops ----
    def use_def(self, def_name: str) -> "ToneView":
        self._b._append(
            UseDefOp(logical_channel=self._logical_channel, def_name=def_name)
        )
        return self

    def move(
        self,
        *,
        df: float,
        time: float,
        idxs: Optional[Sequence[int]] = None,
        kind: str = "linear",
    ) -> "ToneView":
        idx_t = tuple(int(i) for i in idxs) if idxs is not None else None
        self._b._append(
            MoveOp(
                logical_channel=self._logical_channel,
                df_hz=float(df),
                time_s=float(time),
                idxs=idx_t,
                kind=kind,
            )
        )  # type: ignore[arg-type]
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
        amps_t = (
            amps if isinstance(amps, (int, float)) else tuple(float(a) for a in amps)
        )
        self._b._append(
            RampAmpToOp(
                logical_channel=self._logical_channel,
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

        # dst="all" expands to 0..len(target_def)-1 (resolved here by looking up the definition).
        d = self._b._definitions[target_def]
        if dst == "all":
            dst_t = tuple(range(len(d.freqs_hz)))
        else:
            dst_t = tuple(int(i) for i in dst)

        self._b._append(
            RemapFromDefOp(
                logical_channel=self._logical_channel,
                target_def=target_def,
                src=src_t,
                dst=dst_t,
                time_s=float(time),
                kind=kind,  # type: ignore[arg-type]
            )
        )
        return self

    # ---- forward builder methods for fluent chaining ----
    def tones(self, logical_channel: str) -> "ToneView":
        return self._b.tones(logical_channel)

    def segment(
        self,
        name: str,
        mode: SegmentMode = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "carry",
    ) -> "AWGProgramBuilder":
        return self._b.segment(name, mode=mode, loop=loop, phase_mode=phase_mode)

    def hold(
        self, *, time: float, warn_df: Optional[float] = None
    ) -> "AWGProgramBuilder":
        return self._b.hold(time=time, warn_df=warn_df)

    def build(self):
        return self._b.build()


class AWGProgramBuilder:
    """
    Fluent front-end for building an AWG program.

    Notes:
    - A `logical_channel` is just a user-defined name (e.g. "H", "V") selecting which
      tone-bank an operation applies to; it is not a hardware channel.
    - The timeline is continuous: logical-channel state carries across segment
      boundaries; `time=0` ops update state without advancing time.
    """

    def __init__(self, sample_rate: float):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self._fs = float(sample_rate)
        self._logical_channels: List[str] = []
        self._definitions: Dict[str, DefinitionSpec] = {}
        self._segments: List[SegmentSpec] = []
        self._current_seg: Optional[int] = None
        self._calibrations: Dict[str, Any] = {}

    def with_calibration(self, key: str, obj: Any) -> "AWGProgramBuilder":
        self._calibrations[key] = obj
        return self

    def logical_channel(self, name: str) -> "AWGProgramBuilder":
        if name in self._logical_channels:
            return self
        self._logical_channels.append(str(name))
        return self

    def define(
        self,
        name: str,
        *,
        logical_channel: str,
        freqs: Sequence[float],
        amps: Sequence[float],
        phases: str | Sequence[float] = "auto",
    ) -> "AWGProgramBuilder":
        if logical_channel not in self._logical_channels:
            raise ValueError(
                f"Define references unknown logical_channel {logical_channel!r}. "
                f"Call .logical_channel({logical_channel!r}) first."
            )
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
            logical_channel=logical_channel,
            freqs_hz=tuple(float(x) for x in freqs),
            amps=tuple(float(x) for x in amps),
            phases_rad=ph,
        )
        return self

    def segment(
        self,
        name: str,
        mode: SegmentMode = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "carry",
    ) -> "AWGProgramBuilder":
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

        if phase_mode not in ("carry", "fixed"):
            raise ValueError(f"Unknown phase_mode {phase_mode!r}")

        self._segments.append(
            SegmentSpec(
                name=name, mode=mode, loop=int(loop), ops=tuple(), phase_mode=phase_mode
            )
        )
        self._current_seg = len(self._segments) - 1
        return self

    def tones(self, logical_channel: str) -> ToneView:
        if logical_channel not in self._logical_channels:
            raise KeyError(f"Unknown logical_channel {logical_channel!r}")
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .tones(...)")
        return ToneView(self, logical_channel)

    def hold(
        self, *, time: float, warn_df: Optional[float] = None
    ) -> "AWGProgramBuilder":
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .hold(...)")
        self._append(
            HoldOp(
                time_s=float(time),
                warn_df_hz=float(warn_df) if warn_df is not None else None,
            )
        )
        return self

    def _append(self, op) -> None:
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before adding ops")
        seg = self._segments[self._current_seg]
        self._segments[self._current_seg] = SegmentSpec(
            name=seg.name,
            mode=seg.mode,
            loop=seg.loop,
            ops=seg.ops + (op,),
            phase_mode=seg.phase_mode,
        )

    def build_spec(self) -> ProgramSpec:
        if not self._segments:
            raise RuntimeError("No segments defined")
        return ProgramSpec(
            sample_rate_hz=self._fs,
            logical_channels=tuple(self._logical_channels),
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
