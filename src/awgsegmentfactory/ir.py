from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, NewType, Optional, Tuple, Literal

SegmentMode = Literal["loop_n", "wait_trig", "once"]
InterpKind = Literal["hold", "linear", "exp", "min_jerk"]
SegmentPhaseMode = Literal["carry", "fixed"]

ToneId = NewType("ToneId", int)


class PositionToFreqCalib(ABC):
    """
    Calibration interface: convert a requested position delta (e.g. µm) into a
    frequency delta (Hz) for a given tone and logical channel.
    """

    @abstractmethod
    def df_hz(self, tone_id: ToneId, dx_um: float, logical_channel: str) -> float: ...


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    logical_channel: str
    freqs_hz: Tuple[float, ...]
    amps: Tuple[float, ...]
    phases_rad: Tuple[float, ...]  # keep, even if unused now


@dataclass(frozen=True)
class IntentSegment:
    name: str
    mode: SegmentMode
    loop: int  # for loop_n/once. For wait_trig set loop=1.
    ops: Tuple["Op", ...]
    phase_mode: SegmentPhaseMode = "carry"


# ---- Ops ----


class Op:
    pass


@dataclass(frozen=True)
class HoldOp(Op):
    time_s: float
    warn_df_hz: Optional[float] = None  # only meaningful in wait_trig segments


@dataclass(frozen=True)
class UseDefOp(Op):
    logical_channel: str
    def_name: str


@dataclass(frozen=True)
class MoveOp(Op):
    logical_channel: str
    df_hz: float
    time_s: float
    idxs: Optional[Tuple[int, ...]] = None
    kind: InterpKind = "linear"  # "linear" or "min_jerk" etc


@dataclass(frozen=True)
class RampAmpToOp(Op):
    logical_channel: str
    amps_target: float | Tuple[float, ...]
    time_s: float
    kind: InterpKind = "linear"  # "linear" or "exp"
    tau_s: Optional[float] = None
    idxs: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class RemapFromDefOp(Op):
    """
    Retarget: take selected existing tones (src indices in current logical-channel state),
    map onto target definition ordering (dst indices), tween to target def values.
    Typically this also changes the number of tones to len(dst).
    """

    logical_channel: str
    target_def: str
    src: Tuple[int, ...]
    dst: Tuple[int, ...]  # explicit indices into target definition
    time_s: float
    kind: InterpKind = "min_jerk"


@dataclass(frozen=True)
class IntentIR:
    """
    Recorded user intent (continuous-time).

    Notes:
    - Durations are expressed in seconds (`time_s`) and have not yet been converted
      into integer sample counts. That discretization happens in `resolve_intent_ir(...)`.
    """

    logical_channels: Tuple[str, ...]
    definitions: Dict[str, IntentDefinition]
    segments: Tuple[IntentSegment, ...]
    calibrations: Dict[str, object]
