from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Literal

SegmentMode = Literal["loop_n", "wait_trig", "once"]
InterpKind = Literal["hold", "linear", "exp", "min_jerk"]

@dataclass(frozen=True)
class DefinitionSpec:
    name: str
    plane: str
    freqs_hz: Tuple[float, ...]
    amps: Tuple[float, ...]
    phases_rad: Tuple[float, ...]  # keep, even if unused now

@dataclass(frozen=True)
class SegmentSpec:
    name: str
    mode: SegmentMode
    loop: int  # for loop_n/once. For wait_trig set loop=1.
    ops: Tuple["Op", ...]

# ---- Ops ----

class Op:
    pass

@dataclass(frozen=True)
class HoldOp(Op):
    time_s: float
    warn_df_hz: Optional[float] = None  # only meaningful in wait_trig segments

@dataclass(frozen=True)
class UseDefOp(Op):
    plane: str
    def_name: str

@dataclass(frozen=True)
class MoveOp(Op):
    plane: str
    df_hz: float
    time_s: float
    idxs: Optional[Tuple[int, ...]] = None
    kind: InterpKind = "linear"  # "linear" or "min_jerk" etc

@dataclass(frozen=True)
class RampAmpToOp(Op):
    plane: str
    amps_target: float | Tuple[float, ...]
    time_s: float
    kind: InterpKind = "linear"  # "linear" or "exp"
    tau_s: Optional[float] = None
    idxs: Optional[Tuple[int, ...]] = None

@dataclass(frozen=True)
class RemapFromDefOp(Op):
    """
    Retarget: take selected existing tones (src indices in current plane state),
    map onto target definition ordering (dst indices), tween to target def values.
    Typically this also changes the number of tones to len(dst).
    """
    plane: str
    target_def: str
    src: Tuple[int, ...]
    dst: Tuple[int, ...]  # explicit indices into target definition
    time_s: float
    kind: InterpKind = "min_jerk"

@dataclass(frozen=True)
class ProgramSpec:
    sample_rate_hz: float
    planes: Tuple[str, ...]
    definitions: Dict[str, DefinitionSpec]
    segments: Tuple[SegmentSpec, ...]
    calibrations: Dict[str, object]
