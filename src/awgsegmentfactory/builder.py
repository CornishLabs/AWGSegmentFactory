"""Fluent builder for constructing an `IntentIR`.

The builder is the user-facing API. It records operations into a continuous-time
`IntentIR`, which you can then resolve/quantize/compile into hardware-ready samples.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import (
    IntentIR,
    IntentSegment,
    SegmentMode,
    SegmentPhaseMode,
    IntentDefinition,
    HoldOp,
    UseDefOp,
    MoveOp,
    RampAmpToOp,
    RemapFromDefOp,
)
from .program_ir import ResolvedIR
from .resolve import resolve_intent_ir


def _phases_auto(n: int) -> Tuple[float, ...]:
    """Placeholder phase picker used by `AWGProgramBuilder.define(phases="auto")`."""
    # placeholder: you can later implement phase picking for cresting, etc.
    return tuple(0.0 for _ in range(n))


class ToneView:
    """
    View over a particular logical channel (e.g. "H" or "V") within the current segment.
    Implements per-channel ops and forwards builder methods so fluent chaining works.
    """

    def __init__(self, b: "AWGProgramBuilder", logical_channel: str):
        """Create a view bound to one logical channel within the current segment."""
        self._b = b
        self._logical_channel = logical_channel

    # ---- per-logical-channel ops ----
    def use_def(self, def_name: str) -> "ToneView":
        """Append a `UseDefOp` for this logical channel."""
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
        """Append a `MoveOp` (frequency delta over time) for this logical channel."""
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
        """Append a `RampAmpToOp` (amplitude ramp over time) for this logical channel."""
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
        """Append a `RemapFromDefOp` to retarget/select tones against a definition."""
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
        """Switch fluent context to a different logical channel within the same segment."""
        return self._b.tones(logical_channel)

    def segment(
        self,
        name: str,
        mode: SegmentMode = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "carry",
    ) -> "AWGProgramBuilder":
        """Start a new segment and return the builder for further chaining."""
        return self._b.segment(name, mode=mode, loop=loop, phase_mode=phase_mode)

    def hold(
        self, *, time: float, warn_df: Optional[float] = None
    ) -> "AWGProgramBuilder":
        """Append a `HoldOp` (hold all logical channels for a duration)."""
        return self._b.hold(time=time, warn_df=warn_df)

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
        """Define a named tone-bank state used later by `.use_def(...)`."""
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
        mode: SegmentMode = "once",
        loop: Optional[int] = None,
        *,
        phase_mode: SegmentPhaseMode = "carry",
    ) -> "AWGProgramBuilder":
        """Start a new segment; subsequent ops are appended to this segment."""
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
            IntentSegment(
                name=name, mode=mode, loop=int(loop), ops=tuple(), phase_mode=phase_mode
            )
        )
        self._current_seg = len(self._segments) - 1
        return self

    def tones(self, logical_channel: str) -> ToneView:
        """Return a `ToneView` for adding ops targeting `logical_channel`."""
        if logical_channel not in self._logical_channels:
            raise KeyError(f"Unknown logical_channel {logical_channel!r}")
        if self._current_seg is None:
            raise RuntimeError("Call .segment(...) before .tones(...)")
        return ToneView(self, logical_channel)

    def hold(
        self, *, time: float, warn_df: Optional[float] = None
    ) -> "AWGProgramBuilder":
        """Append a `HoldOp` to the current segment (holds *all* logical channels)."""
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
        """Append an op to the current segment (internal helper for `ToneView`)."""
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
