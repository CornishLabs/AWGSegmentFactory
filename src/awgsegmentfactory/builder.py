"""Fluent builder for constructing an `IntentIR`.

The builder is the user-facing API. It records operations into a continuous-time
`IntentIR`, which you can then resolve/quantize/compile into hardware-ready samples.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .intent_ir import (
    IntentIR,
    IntentSegment,
    InterpKind,
    InterpSpec,
    SegmentMode,
    SegmentPhaseMode,
    IntentDefinition,
    HoldOp,
    UseDefOp,
    AddToneOp,
    RemoveTonesOp,
    ParallelOp,
    MoveOp,
    RampAmpToOp,
    RemapFromDefOp,
)
from .resolved_ir import ResolvedIR
from .resolve import resolve_intent_ir


def _phases_auto(n: int) -> Tuple[float, ...]:
    """
    Default phase picker used by `AWGProgramBuilder.define(phases="auto")`.

    Currently returns all zeros. If a segment uses `phase_mode="optimise"` or
    `phase_mode="continue"`, start phases may be overridden during sample synthesis.
    """
    return tuple(0.0 for _ in range(n))


SegmentModeArg = SegmentMode | Literal["once"]


class LogicalChannelView:
    """
    View over a particular logical channel (e.g. "H" or "V") within the current segment.
    Implements per-channel ops and forwards builder methods so fluent chaining works.
    """

    def __init__(self, b: "AWGProgramBuilder", logical_channel: str):
        """Create a view bound to one logical channel within the current segment."""
        self._b = b
        self._logical_channel = logical_channel

    # ---- per-logical-channel ops ----
    def use_def(self, def_name: str) -> "LogicalChannelView":
        """Append a `UseDefOp` for this logical channel."""
        self._b._append(
            UseDefOp(logical_channel=self._logical_channel, def_name=def_name)
        )
        return self

    def add_tone(
        self,
        *,
        f: float,
        amp: float = 0.0,
        phase: float | Literal["auto"] = "auto",
        at: Optional[int] = None,
    ) -> "LogicalChannelView":
        """Append an `AddToneOp` (instantaneous tone-bank edit) for this logical channel."""
        ph = 0.0 if phase == "auto" else float(phase)
        self._b._append(
            AddToneOp(
                logical_channel=self._logical_channel,
                freqs_hz=(float(f),),
                amps=(float(amp),),
                phases_rad=(ph,),
                at=int(at) if at is not None else None,
            )
        )
        return self

    def remove_tones(self, *, idxs: Sequence[int]) -> "LogicalChannelView":
        """Append a `RemoveTonesOp` (instantaneous tone-bank edit) for this logical channel."""
        idx_t = tuple(int(i) for i in idxs)
        if not idx_t:
            raise ValueError("remove_tones: idxs must be non-empty")
        self._b._append(
            RemoveTonesOp(logical_channel=self._logical_channel, idxs=idx_t)
        )
        return self

    def remove_tone(self, *, idx: int) -> "LogicalChannelView":
        """Convenience: remove a single tone by index."""
        return self.remove_tones(idxs=[int(idx)])

    def move(
        self,
        *,
        df: float,
        time: float,
        idxs: Optional[Sequence[int]] = None,
        kind: InterpKind = "linear",
    ) -> "LogicalChannelView":
        """Append a `MoveOp` (frequency delta over time) for this logical channel."""
        if kind not in ("linear", "min_jerk"):
            raise ValueError("move: kind must be 'linear' or 'min_jerk'")
        idx_t = tuple(int(i) for i in idxs) if idxs is not None else None
        self._b._append(
            MoveOp(
                logical_channel=self._logical_channel,
                df_hz=float(df),
                time_s=float(time),
                idxs=idx_t,
                interp=InterpSpec(kind),
            )
        )
        return self

    def ramp_amp_to(
        self,
        *,
        amps: float | Sequence[float],
        time: float,
        kind: InterpKind = "linear",
        tau: Optional[float] = None,
        idxs: Optional[Sequence[int]] = None,
    ) -> "LogicalChannelView":
        """Append a `RampAmpToOp` (amplitude ramp over time) for this logical channel."""
        if kind not in ("linear", "exp", "min_jerk", "geo_ramp", "adiabatic_ramp"):
            raise ValueError(
                "ramp_amp_to: kind must be one of "
                "'linear', 'exp', 'min_jerk', 'geo_ramp', 'adiabatic_ramp'"
            )
        if kind == "exp":
            if tau is None:
                raise ValueError("ramp_amp_to: kind='exp' requires tau=...")
            interp = InterpSpec(kind, tau_s=float(tau))
        else:
            if tau is not None:
                raise ValueError("ramp_amp_to: tau is only valid for kind='exp'")
            interp = InterpSpec(kind)
        idx_t = tuple(int(i) for i in idxs) if idxs is not None else None
        amps_t = (
            amps if isinstance(amps, (int, float)) else tuple(float(a) for a in amps)
        )
        self._b._append(
            RampAmpToOp(
                logical_channel=self._logical_channel,
                amps_target=amps_t,  # type: ignore[arg-type]
                time_s=float(time),
                interp=interp,
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
        kind: InterpKind = "min_jerk",
    ) -> "LogicalChannelView":
        """Append a `RemapFromDefOp` to retarget/select tones against a definition."""
        if kind not in ("linear", "min_jerk"):
            raise ValueError("remap_from_def: kind must be 'linear' or 'min_jerk'")
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
                interp=InterpSpec(kind),
            )
        )
        return self

    # ---- forward builder methods for fluent chaining ----
    def tones(self, logical_channel: str) -> "LogicalChannelView":
        """Switch fluent context to a different logical channel within the same segment."""
        return self._b.tones(logical_channel)

    def segment(
        self,
        name: str,
        mode: SegmentModeArg = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "continue",
    ) -> "AWGProgramBuilder":
        """Start a new segment and return the builder for further chaining."""
        return self._b.segment(name, mode=mode, loop=loop, phase_mode=phase_mode)

    def hold(self, *, time: float) -> "AWGProgramBuilder":
        """Append a `HoldOp` (hold all logical channels for a duration)."""
        return self._b.hold(time=time)

    def build(self):
        """Disallow `.build()` to avoid ambiguity; use `build_*` methods instead."""
        raise AttributeError(
            "Use build_intent_ir(), build_resolved_ir(...), or build_timeline(...) on the builder."
        )

    def build_intent_ir(self) -> IntentIR:
        """Return the continuous-time `IntentIR` recorded by this builder."""
        return self._b.build_intent_ir()

    def build_resolved_ir(self, *, sample_rate_hz: float) -> ResolvedIR:
        """Resolve intent into integer-sample primitives (`ResolvedIR`)."""
        return self._b.build_resolved_ir(sample_rate_hz=sample_rate_hz)

    def build_timeline(self, *, sample_rate_hz: float):
        """Convenience: resolve intent and convert to a debug `ResolvedTimeline`."""
        return self._b.build_timeline(sample_rate_hz=sample_rate_hz)


class _ParallelBlockBuilder:
    """Internal helper for building a `ParallelOp`."""

    def __init__(self, parent: "AWGProgramBuilder"):
        self._parent = parent
        self._definitions = parent._definitions
        self._ops: list[Any] = []

    def tones(self, logical_channel: str) -> LogicalChannelView:
        if logical_channel not in self._parent._logical_channels:
            raise KeyError(f"Unknown logical_channel {logical_channel!r}")
        return LogicalChannelView(self, logical_channel)

    def segment(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("parallel: cannot start a new segment inside a parallel block")

    def hold(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("parallel: use .hold(...) outside the parallel block")

    def _append(self, op) -> None:
        self._ops.append(op)


class AWGProgramBuilder:
    """
    Fluent front-end for building an AWG program.

    Notes:
    - A `logical_channel` is just a user-defined name (e.g. "H", "V") selecting which
      tone-bank an operation applies to; it is not a hardware channel.
    - The timeline is continuous: logical-channel state carries across segment
      boundaries; `time=0` ops update state without advancing time.
    """

    def __init__(self):
        """Create an empty builder with no logical channels, definitions, or segments."""
        self._logical_channels: List[str] = []
        self._definitions: Dict[str, IntentDefinition] = {}
        self._segments: List[IntentSegment] = []
        self._current_seg: Optional[int] = None
        self._calibrations: Dict[str, Any] = {}

    def with_calibration(self, key: str, obj: Any) -> "AWGProgramBuilder":
        """Attach an arbitrary calibration object to the resulting `IntentIR`."""
        self._calibrations[key] = obj
        return self

    def logical_channel(self, name: str) -> "AWGProgramBuilder":
        """Register a logical channel name (e.g. "H", "V") used by ops/definitions."""
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
        """
        Define a named tone-bank state used later by `.use_def(...)`.

        Notes:
        - `phases="auto"` currently means all phases are set to 0.
        - These phases are used directly when a segment uses `phase_mode="manual"`.
          Other phase modes may override start phases during compilation.
        """
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

        self._definitions[name] = IntentDefinition(
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
        mode: SegmentModeArg = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "continue",
    ) -> "AWGProgramBuilder":
        """
        Start a new segment; subsequent ops are appended to this segment.

        `phase_mode` controls how *start phases* are chosen during sample synthesis:
        - `"manual"`: use the phases stored in the IR.
        - `"optimise"`: crest-optimise all start phases from the segment's start freqs/amps.
        - `"continue"`: continue matching tone phases from the previous segment (by frequency),
          and crest-optimise any new/unmatched tones while keeping continued tones fixed.

        Note: phase optimisation happens at compile time (`compile_sequence_program(...)`),
        not in `ResolvedIR.to_timeline()`.
        """
        mode_s = str(mode)
        resolved_mode: SegmentMode
        if mode_s == "once":
            resolved_mode = "loop_n"
            loop = 1
        elif mode_s == "loop_n":
            resolved_mode = "loop_n"
            if loop is None or loop <= 0:
                raise ValueError("loop_n requires loop=N with N>0")
        elif mode_s == "wait_trig":
            resolved_mode = "wait_trig"
            loop = 1
        else:
            raise ValueError(f"Unknown mode {mode_s!r}")

        if phase_mode not in ("manual", "continue", "optimise"):
            raise ValueError("phase_mode must be one of 'manual', 'continue', 'optimise'")

        self._segments.append(
            IntentSegment(
                name=name,
                mode=resolved_mode,
                loop=int(loop),
                ops=tuple(),
                phase_mode=phase_mode,
            )
        )
        self._current_seg = len(self._segments) - 1
        return self

    def tones(self, logical_channel: str) -> LogicalChannelView:
        """Return a `LogicalChannelView` for adding ops targeting `logical_channel`."""
        if logical_channel not in self._logical_channels:
            raise KeyError(f"Unknown logical_channel {logical_channel!r}")
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .tones(...)")
        return LogicalChannelView(self, logical_channel)

    def hold(self, *, time: float) -> "AWGProgramBuilder":
        """Append a `HoldOp` to the current segment (holds *all* logical channels)."""
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .hold(...)")
        self._append(
            HoldOp(
                time_s=float(time),
            )
        )
        return self

    def parallel(self, fn) -> "AWGProgramBuilder":
        """
        Collect per-channel ops and run them concurrently as one time-interval.

        The function receives a restricted builder that supports `.tones(...).move(...)`,
        `.tones(...).ramp_amp_to(...)`, and `.tones(...).remap_from_def(...)`.
        """
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .parallel(...)")

        block = _ParallelBlockBuilder(self)
        fn(block)
        if not block._ops:
            raise ValueError("parallel: no ops recorded")

        timed_ops: list[Any] = []
        for op in block._ops:
            if isinstance(op, (MoveOp, RampAmpToOp, RemapFromDefOp)) and op.time_s > 0:
                timed_ops.append(op)
            else:
                self._append(op)

        if not timed_ops:
            return self

        time_s = float(timed_ops[0].time_s)
        for op in timed_ops[1:]:
            if not math.isclose(float(op.time_s), time_s, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("parallel: all timed ops must have the same time=...")

        logical_channels: set[str] = set()
        for op in timed_ops:
            if op.logical_channel in logical_channels:
                raise ValueError(
                    f"parallel: multiple timed ops for logical_channel {op.logical_channel!r}"
                )
            logical_channels.add(op.logical_channel)

        self._append(ParallelOp(time_s=time_s, ops=tuple(timed_ops)))
        return self

    def _append(self, op) -> None:
        """Append an op to the current segment (internal helper for `LogicalChannelView`)."""
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before adding ops")
        seg = self._segments[self._current_seg]
        self._segments[self._current_seg] = IntentSegment(
            name=seg.name,
            mode=seg.mode,
            loop=seg.loop,
            ops=seg.ops + (op,),
            phase_mode=seg.phase_mode,
        )

    def build_intent_ir(self) -> IntentIR:
        """Finalize and return the recorded `IntentIR` without resolving to samples."""
        if not self._segments:
            raise RuntimeError("No segments defined")
        return IntentIR(
            logical_channels=tuple(self._logical_channels),
            definitions=dict(self._definitions),
            segments=tuple(self._segments),
            calibrations=dict(self._calibrations),
        )

    def build_timeline(self, *, sample_rate_hz: float):
        """Resolve intent and return a debug-friendly `ResolvedTimeline`."""
        return self.build_resolved_ir(sample_rate_hz=sample_rate_hz).to_timeline()

    def build_resolved_ir(self, *, sample_rate_hz: float) -> ResolvedIR:
        """Resolve intent into `ResolvedIR` (integer-sample, segment-grouped primitives)."""
        return resolve_intent_ir(self.build_intent_ir(), sample_rate_hz=sample_rate_hz)
