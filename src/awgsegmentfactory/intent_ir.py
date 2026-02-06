"""Intent IR dataclasses (continuous-time program specification).

This module defines the "what the user wants" representation produced by the builder.
It is named `intent_ir` to distinguish it from the later `resolved_ir` stage.
It is later resolved into integer-sample primitives by `resolve_intent_ir(...)`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from typing import Dict, NewType, Optional, Tuple, Literal

SegmentMode = Literal["loop_n", "wait_trig"]
InterpKind = Literal["hold", "linear", "exp", "min_jerk", "geo_ramp", "adiabatic_ramp"]
SegmentPhaseMode = Literal["manual", "continue", "optimise"]

ToneId = NewType("ToneId", int)

_ENCODED_INTENT_IR_SCHEMA = "awgsegmentfactory.intent_ir.IntentIR"
_ENCODED_INTENT_IR_VERSION = 1

@dataclass(frozen=True)
class InterpSpec:
    """Interpolation spec (kind + kind-specific parameters).

    This avoids scattering optional interpolation parameters (like `tau_s` for `"exp"`)
    across multiple ops/IR layers.
    """

    kind: InterpKind
    tau_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind == "exp":
            if self.tau_s is None:
                raise ValueError("InterpSpec(kind='exp') requires tau_s")
        else:
            if self.tau_s is not None:
                raise ValueError("tau_s is only valid for kind='exp'")


class PositionToFreqCalib(ABC):
    """
    Calibration interface for turning a position delta into a frequency delta.

    Intended for higher-level ops (e.g. "move by +dx") before becoming low-level
    frequency changes.
    """

    @abstractmethod
    def df_hz(self, tone_id: ToneId, dx_um: float, logical_channel: str) -> float:
        """Return frequency delta in Hz for a requested position delta in µm."""
        raise NotImplementedError


@dataclass(frozen=True)
class IntentDefinition:
    """Named initial tone-bank state for a logical channel (freq/amp/phase arrays)."""

    name: str
    logical_channel: str
    freqs_hz: Tuple[float, ...]
    amps: Tuple[float, ...]
    # Used when segment `phase_mode="manual"`. Other phase modes may override start phases
    # during sample synthesis (compile time).
    phases_rad: Tuple[float, ...]


@dataclass(frozen=True)
class IntentSegment:
    """One user-defined segment: a sequence of ops plus sequencing metadata."""

    name: str
    mode: SegmentMode
    loop: int  # for loop_n. For wait_trig set loop=1.
    ops: Tuple["Op", ...]
    phase_mode: SegmentPhaseMode = "continue"
    # Quantization preferences (applied in `quantize_resolved_ir`):
    # - `snap_len_to_quantum`: for loopable segments, snap length to the global quantum
    #   (reduces wrap-snapping error but increases trigger/loop latency).
    # - `snap_freqs_to_wrap`: for constant, loopable segments, adjust freqs so the segment
    #   wraps phase-continuously at its quantized length.
    snap_len_to_quantum: bool = True
    snap_freqs_to_wrap: bool = True


# ---- Ops ----


class Op:
    """Base class for intent operations inside an `IntentSegment`."""

    pass


@dataclass(frozen=True)
class HoldOp(Op):
    """Hold the current state for `time_s` seconds (no parameter changes)."""

    time_s: float


@dataclass(frozen=True)
class UseDefOp(Op):
    """Reset a logical channel's tone-bank state to a named `IntentDefinition`."""

    logical_channel: str
    def_name: str


@dataclass(frozen=True)
class AddToneOp(Op):
    """Add one or more tones to a logical channel (instantaneous state update)."""

    logical_channel: str
    freqs_hz: Tuple[float, ...]
    amps: Tuple[float, ...]
    phases_rad: Tuple[float, ...]
    at: Optional[int] = None  # insertion index (None -> append)


@dataclass(frozen=True)
class RemoveTonesOp(Op):
    """Remove tones by index from a logical channel (instantaneous state update)."""

    logical_channel: str
    idxs: Tuple[int, ...]


@dataclass(frozen=True)
class ParallelOp(Op):
    """Run multiple per-logical-channel ops concurrently over `time_s`."""

    time_s: float
    ops: Tuple["Op", ...]


@dataclass(frozen=True)
class MoveOp(Op):
    """Add `df_hz` to selected tone frequencies over `time_s` using `interp`."""

    logical_channel: str
    df_hz: float
    time_s: float
    idxs: Optional[Tuple[int, ...]] = None
    interp: InterpSpec = field(default_factory=lambda: InterpSpec("linear"))


@dataclass(frozen=True)
class RampAmpToOp(Op):
    """Ramp amplitudes to a target value(s) over `time_s` using `interp`."""

    logical_channel: str
    amps_target: float | Tuple[float, ...]
    time_s: float
    interp: InterpSpec = field(default_factory=lambda: InterpSpec("linear"))
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
    interp: InterpSpec = field(default_factory=lambda: InterpSpec("min_jerk"))


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

    def encode(self) -> dict[str, object]:
        """
        Encode this `IntentIR` into a Python dictionary.

        Notes:
        - The output is pure-Python (dict/list/tuple/str/int/float/bool/None).
        - This is intended for Python-to-Python transport (e.g. via pickle/msgpack).
        """
        return {
            "__schema__": _ENCODED_INTENT_IR_SCHEMA,
            "version": _ENCODED_INTENT_IR_VERSION,
            "logical_channels": tuple(str(x) for x in self.logical_channels),
            "definitions": {
                str(k): _encode_intent_definition(v) for k, v in self.definitions.items()
            },
            "segments": tuple(_encode_intent_segment(s) for s in self.segments),
        }

    @classmethod
    def decode(cls, data: dict[str, object]) -> "IntentIR":
        """Decode a dictionary previously produced by `IntentIR.encode()`."""
        if not isinstance(data, dict):
            raise TypeError("IntentIR.decode: data must be a dict")

        schema = data.get("__schema__")
        version = data.get("version")
        if schema != _ENCODED_INTENT_IR_SCHEMA:
            raise ValueError(
                f"IntentIR.decode: unsupported schema {schema!r} "
                f"(expected {_ENCODED_INTENT_IR_SCHEMA!r})"
            )
        if version != _ENCODED_INTENT_IR_VERSION:
            raise ValueError(
                f"IntentIR.decode: unsupported version {version!r} "
                f"(expected {_ENCODED_INTENT_IR_VERSION})"
            )

        logical_channels_raw = data.get("logical_channels", ())
        if isinstance(logical_channels_raw, tuple):
            logical_channels_seq = logical_channels_raw
        elif isinstance(logical_channels_raw, list):
            logical_channels_seq = tuple(logical_channels_raw)
        else:
            raise TypeError("IntentIR.decode: logical_channels must be a list/tuple")
        logical_channels = tuple(str(x) for x in logical_channels_seq)

        defs_raw = data.get("definitions", {})
        if not isinstance(defs_raw, dict):
            raise TypeError("IntentIR.decode: definitions must be a dict")
        definitions = {
            str(k): _decode_intent_definition(_require_dict(v, f"definitions[{k!r}]"))
            for k, v in defs_raw.items()
        }

        segs_raw = data.get("segments", ())
        if isinstance(segs_raw, tuple):
            segs_seq = segs_raw
        elif isinstance(segs_raw, list):
            segs_seq = tuple(segs_raw)
        else:
            raise TypeError("IntentIR.decode: segments must be a list/tuple")
        segments = tuple(
            _decode_intent_segment(_require_dict(s, "segments[i]")) for s in segs_seq
        )

        return cls(
            logical_channels=logical_channels,
            definitions=definitions,
            segments=segments,
        )

    def digest(self, *, algo: str = "sha256") -> str:
        """
        Deterministic hash of this `IntentIR` (content only).

        This can be used as a cache key to avoid re-sending unchanged programs.
        """
        payload = {
            "logical_channels": tuple(str(x) for x in self.logical_channels),
            "definitions": {
                str(k): _encode_intent_definition(v) for k, v in self.definitions.items()
            },
            "segments": tuple(_encode_intent_segment(s) for s in self.segments),
        }
        b = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.new(str(algo), b).hexdigest()

    def segment_fingerprints(
        self,
        *,
        algo: str = "sha256",
        include_referenced_definitions: bool = True,
    ) -> tuple[dict[str, object], ...]:
        """
        Return per-segment fingerprints suitable for incremental syncing.

        Each entry has:
        - `index`: segment index in program order
        - `name`: segment name
        - `digest`: hex digest of the segment (and optionally referenced definitions)
        """
        out: list[dict[str, object]] = []
        for i, seg in enumerate(self.segments):
            payload: dict[str, object] = {
                "segment": _encode_intent_segment(seg),
            }
            if include_referenced_definitions:
                refs = sorted(_referenced_def_names(seg))
                payload["definitions"] = {
                    name: _encode_intent_definition(self.definitions[name])
                    for name in refs
                }

            b = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            digest = hashlib.new(str(algo), b).hexdigest()
            out.append({"index": i, "name": str(seg.name), "digest": digest})
        return tuple(out)


def _require_dict(obj: object, name: str) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be a dict, got {type(obj).__name__}")
    return obj


def _encode_interp_spec(interp: InterpSpec) -> dict[str, object]:
    return {
        "kind": str(interp.kind),
        "tau_s": None if interp.tau_s is None else float(interp.tau_s),
    }


def _decode_interp_spec(data: dict[str, object]) -> InterpSpec:
    kind = data.get("kind")
    if not isinstance(kind, str):
        raise TypeError("interp.kind must be a string")
    tau_s = data.get("tau_s", None)
    if tau_s is not None:
        tau_s = float(tau_s)
    return InterpSpec(kind, tau_s=tau_s)


def _encode_intent_definition(d: IntentDefinition) -> dict[str, object]:
    return {
        "name": str(d.name),
        "logical_channel": str(d.logical_channel),
        "freqs_hz": tuple(float(x) for x in d.freqs_hz),
        "amps": tuple(float(x) for x in d.amps),
        "phases_rad": tuple(float(x) for x in d.phases_rad),
    }


def _decode_intent_definition(data: dict[str, object]) -> IntentDefinition:
    freqs_raw = data.get("freqs_hz", ())
    amps_raw = data.get("amps", ())
    phases_raw = data.get("phases_rad", ())
    freqs = tuple(float(x) for x in (freqs_raw if isinstance(freqs_raw, (list, tuple)) else ()))
    amps = tuple(float(x) for x in (amps_raw if isinstance(amps_raw, (list, tuple)) else ()))
    phases = tuple(float(x) for x in (phases_raw if isinstance(phases_raw, (list, tuple)) else ()))
    return IntentDefinition(
        name=str(data.get("name")),
        logical_channel=str(data.get("logical_channel")),
        freqs_hz=freqs,
        amps=amps,
        phases_rad=phases,
    )


def _encode_intent_segment(seg: IntentSegment) -> dict[str, object]:
    return {
        "name": str(seg.name),
        "mode": str(seg.mode),
        "loop": int(seg.loop),
        "phase_mode": str(seg.phase_mode),
        "snap_len_to_quantum": bool(seg.snap_len_to_quantum),
        "snap_freqs_to_wrap": bool(seg.snap_freqs_to_wrap),
        "ops": tuple(_encode_op(op) for op in seg.ops),
    }


def _decode_intent_segment(data: dict[str, object]) -> IntentSegment:
    ops_raw = data.get("ops", ())
    if isinstance(ops_raw, tuple):
        ops_seq = ops_raw
    elif isinstance(ops_raw, list):
        ops_seq = tuple(ops_raw)
    else:
        raise TypeError("segment.ops must be a list/tuple")
    ops = tuple(_decode_op(_require_dict(op, "segment.ops[i]")) for op in ops_seq)
    return IntentSegment(
        name=str(data.get("name")),
        mode=str(data.get("mode")),
        loop=int(data.get("loop")),  # type: ignore[arg-type]
        ops=ops,
        phase_mode=str(data.get("phase_mode", "continue")),
        snap_len_to_quantum=bool(data.get("snap_len_to_quantum", True)),
        snap_freqs_to_wrap=bool(data.get("snap_freqs_to_wrap", True)),
    )


def _encode_op(op: Op) -> dict[str, object]:
    if isinstance(op, HoldOp):
        return {"op": "HoldOp", "time_s": float(op.time_s)}
    if isinstance(op, UseDefOp):
        return {
            "op": "UseDefOp",
            "logical_channel": str(op.logical_channel),
            "def_name": str(op.def_name),
        }
    if isinstance(op, AddToneOp):
        return {
            "op": "AddToneOp",
            "logical_channel": str(op.logical_channel),
            "freqs_hz": tuple(float(x) for x in op.freqs_hz),
            "amps": tuple(float(x) for x in op.amps),
            "phases_rad": tuple(float(x) for x in op.phases_rad),
            "at": None if op.at is None else int(op.at),
        }
    if isinstance(op, RemoveTonesOp):
        return {
            "op": "RemoveTonesOp",
            "logical_channel": str(op.logical_channel),
            "idxs": tuple(int(i) for i in op.idxs),
        }
    if isinstance(op, ParallelOp):
        return {
            "op": "ParallelOp",
            "time_s": float(op.time_s),
            "ops": tuple(_encode_op(x) for x in op.ops),
        }
    if isinstance(op, MoveOp):
        return {
            "op": "MoveOp",
            "logical_channel": str(op.logical_channel),
            "df_hz": float(op.df_hz),
            "time_s": float(op.time_s),
            "idxs": None if op.idxs is None else tuple(int(i) for i in op.idxs),
            "interp": _encode_interp_spec(op.interp),
        }
    if isinstance(op, RampAmpToOp):
        amps_target: object
        if isinstance(op.amps_target, tuple):
            amps_target = tuple(float(x) for x in op.amps_target)
        else:
            amps_target = float(op.amps_target)
        return {
            "op": "RampAmpToOp",
            "logical_channel": str(op.logical_channel),
            "amps_target": amps_target,
            "time_s": float(op.time_s),
            "interp": _encode_interp_spec(op.interp),
            "idxs": None if op.idxs is None else tuple(int(i) for i in op.idxs),
        }
    if isinstance(op, RemapFromDefOp):
        return {
            "op": "RemapFromDefOp",
            "logical_channel": str(op.logical_channel),
            "target_def": str(op.target_def),
            "src": tuple(int(i) for i in op.src),
            "dst": tuple(int(i) for i in op.dst),
            "time_s": float(op.time_s),
            "interp": _encode_interp_spec(op.interp),
        }
    raise TypeError(f"Unsupported op type: {type(op).__name__}")


def _decode_op(data: dict[str, object]) -> Op:
    op_name = data.get("op")
    if not isinstance(op_name, str):
        raise TypeError("op.op must be a string")

    if op_name == "HoldOp":
        return HoldOp(time_s=float(data.get("time_s")))  # type: ignore[arg-type]
    if op_name == "UseDefOp":
        return UseDefOp(
            logical_channel=str(data.get("logical_channel")),
            def_name=str(data.get("def_name")),
        )
    if op_name == "AddToneOp":
        freqs_raw = data.get("freqs_hz", ())
        amps_raw = data.get("amps", ())
        phases_raw = data.get("phases_rad", ())
        freqs = tuple(float(x) for x in (freqs_raw if isinstance(freqs_raw, (list, tuple)) else ()))
        amps = tuple(float(x) for x in (amps_raw if isinstance(amps_raw, (list, tuple)) else ()))
        phases = tuple(float(x) for x in (phases_raw if isinstance(phases_raw, (list, tuple)) else ()))
        at = data.get("at", None)
        return AddToneOp(
            logical_channel=str(data.get("logical_channel")),
            freqs_hz=freqs,
            amps=amps,
            phases_rad=phases,
            at=None if at is None else int(at),  # type: ignore[arg-type]
        )
    if op_name == "RemoveTonesOp":
        idxs_raw = data.get("idxs", ())
        if not isinstance(idxs_raw, (list, tuple)):
            raise TypeError("RemoveTonesOp.idxs must be a list/tuple")
        return RemoveTonesOp(
            logical_channel=str(data.get("logical_channel")),
            idxs=tuple(int(i) for i in idxs_raw),
        )
    if op_name == "ParallelOp":
        ops_raw = data.get("ops", ())
        if isinstance(ops_raw, tuple):
            ops_seq = ops_raw
        elif isinstance(ops_raw, list):
            ops_seq = tuple(ops_raw)
        else:
            raise TypeError("ParallelOp.ops must be a list/tuple")
        ops = tuple(_decode_op(_require_dict(o, "ParallelOp.ops[i]")) for o in ops_seq)
        return ParallelOp(time_s=float(data.get("time_s")), ops=ops)  # type: ignore[arg-type]
    if op_name == "MoveOp":
        idxs_raw = data.get("idxs", None)
        if idxs_raw is None:
            idxs = None
        else:
            if not isinstance(idxs_raw, (list, tuple)):
                raise TypeError("MoveOp.idxs must be a list/tuple or None")
            idxs = tuple(int(i) for i in idxs_raw)
        return MoveOp(
            logical_channel=str(data.get("logical_channel")),
            df_hz=float(data.get("df_hz")),  # type: ignore[arg-type]
            time_s=float(data.get("time_s")),  # type: ignore[arg-type]
            idxs=idxs,
            interp=_decode_interp_spec(_require_dict(data.get("interp"), "interp")),
        )
    if op_name == "RampAmpToOp":
        amps_target_raw = data.get("amps_target")
        if isinstance(amps_target_raw, (list, tuple)):
            amps_target: float | Tuple[float, ...] = tuple(
                float(x) for x in amps_target_raw
            )
        else:
            amps_target = float(amps_target_raw)  # type: ignore[arg-type]
        idxs_raw = data.get("idxs", None)
        if idxs_raw is None:
            idxs = None
        else:
            if not isinstance(idxs_raw, (list, tuple)):
                raise TypeError("RampAmpToOp.idxs must be a list/tuple or None")
            idxs = tuple(int(i) for i in idxs_raw)
        return RampAmpToOp(
            logical_channel=str(data.get("logical_channel")),
            amps_target=amps_target,
            time_s=float(data.get("time_s")),  # type: ignore[arg-type]
            interp=_decode_interp_spec(_require_dict(data.get("interp"), "interp")),
            idxs=idxs,
        )
    if op_name == "RemapFromDefOp":
        src_raw = data.get("src", ())
        dst_raw = data.get("dst", ())
        if not isinstance(src_raw, (list, tuple)) or not isinstance(dst_raw, (list, tuple)):
            raise TypeError("RemapFromDefOp.src/dst must be list/tuple")
        return RemapFromDefOp(
            logical_channel=str(data.get("logical_channel")),
            target_def=str(data.get("target_def")),
            src=tuple(int(i) for i in src_raw),
            dst=tuple(int(i) for i in dst_raw),
            time_s=float(data.get("time_s")),  # type: ignore[arg-type]
            interp=_decode_interp_spec(_require_dict(data.get("interp"), "interp")),
        )

    raise ValueError(f"Unknown op {op_name!r}")


def _referenced_def_names(seg: IntentSegment) -> set[str]:
    """Return the set of definition names referenced by ops inside a segment."""
    refs: set[str] = set()

    def walk(op: Op) -> None:
        if isinstance(op, UseDefOp):
            refs.add(str(op.def_name))
            return
        if isinstance(op, RemapFromDefOp):
            refs.add(str(op.target_def))
            return
        if isinstance(op, ParallelOp):
            for child in op.ops:
                walk(child)
            return

    for op in seg.ops:
        walk(op)
    return refs
