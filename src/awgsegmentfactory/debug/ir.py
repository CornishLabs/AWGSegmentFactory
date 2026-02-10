"""Text formatting helpers for inspecting IR objects.

These helpers are intentionally lightweight (no matplotlib) so they can be used in
simple scripts and unit tests.
"""

from __future__ import annotations

from typing import Iterable, Optional, TypeVar

from ..builder import AWGProgramBuilder
from ..intent_ir import IntentIR, InterpSpec
from ..quantize import QuantizedIR, format_samples_time
from ..resolved_ir import ResolvedIR


def _fmt_interp(interp: InterpSpec, *, include_params: bool) -> str:
    if not include_params:
        return str(interp.kind)
    if interp.kind == "exp" and interp.tau_s is not None:
        return f"exp(tau_s={interp.tau_s:g})"
    return str(interp.kind)


T = TypeVar("T")


def _limit(items: Iterable[T], *, max_items: Optional[int]) -> tuple[list[T], int]:
    out: list[T] = []
    hidden = 0
    for i, x in enumerate(items):
        if max_items is not None and i >= max_items:
            hidden += 1
            continue
        out.append(x)
    return out, hidden


def format_intent_ir(
    intent: IntentIR,
    *,
    max_segments: Optional[int] = None,
    max_ops_per_segment: Optional[int] = None,
) -> str:
    """Format an `IntentIR` as a readable multi-line summary."""
    lines: list[str] = []
    defs = list(intent.definitions.keys())
    lines.append(
        "IntentIR: "
        f"segments={len(intent.segments)} "
        f"logical_channels={intent.logical_channels} "
        f"definitions={defs}"
    )

    segs, hidden_segs = _limit(intent.segments, max_items=max_segments)
    for seg in segs:
        op_names = (type(op).__name__ for op in seg.ops)
        ops, hidden_ops = _limit(op_names, max_items=max_ops_per_segment)
        ops_desc = f"[{', '.join(str(x) for x in ops)}]"
        if hidden_ops:
            ops_desc = ops_desc[:-1] + f", …(+{hidden_ops})]"
        lines.append(
            f"- segment {seg.name!r}: mode={seg.mode} loop={seg.loop} "
            f"phase_mode={seg.phase_mode} ops={ops_desc}"
        )

    if hidden_segs:
        lines.append(f"- …(+{hidden_segs} more segments)")
    return "\n".join(lines)


def format_resolved_ir(
    ir: ResolvedIR,
    *,
    unit: str = "us",
    logical_channels: Optional[Iterable[str]] = None,
    max_segments: Optional[int] = None,
    max_parts_per_segment: Optional[int] = None,
    include_phase_mode: bool = True,
    include_quantize_prefs: bool = False,
    include_interp_params: bool = False,
    include_tone_counts: bool = False,
) -> str:
    """Format a `ResolvedIR` as a readable multi-line summary."""
    fs = float(ir.sample_rate_hz)
    lcs = tuple(str(x) for x in (logical_channels or ir.logical_channels))

    lines: list[str] = []
    lines.append(
        "ResolvedIR: "
        f"segments={len(ir.segments)} "
        f"total_samples={ir.n_samples} "
        f"duration_s={ir.duration_s:.6g} "
        f"fs_hz={fs:.6g} "
        f"logical_channels={lcs}"
    )

    segs, hidden_segs = _limit(ir.segments, max_items=max_segments)
    for seg in segs:
        seg_meta: list[str] = [f"mode={seg.mode}", f"loop={seg.loop}"]
        if include_phase_mode:
            seg_meta.append(f"phase_mode={seg.phase_mode}")
        if include_quantize_prefs:
            seg_meta.append(f"snap_len_to_quantum={seg.snap_len_to_quantum}")
            seg_meta.append(f"snap_freqs_to_wrap={seg.snap_freqs_to_wrap}")
        seg_meta.append(f"samples={seg.n_samples}")
        seg_meta.append(f"parts={len(seg.parts)}")
        lines.append(f"- segment {seg.name!r}: " + " ".join(seg_meta))

        shown_parts = seg.parts
        hidden_parts = 0
        if max_parts_per_segment is not None and len(seg.parts) > max_parts_per_segment:
            shown_parts = seg.parts[:max_parts_per_segment]
            hidden_parts = len(seg.parts) - max_parts_per_segment

        for i, part in enumerate(shown_parts):
            part_desc = f"{format_samples_time(part.n_samples, fs, unit=unit)}"
            lc_descs: list[str] = []
            for lc in lcs:
                pp = part.logical_channels.get(lc)
                if pp is None:
                    lc_descs.append(f"{lc}:—")
                    continue
                interp_desc = _fmt_interp(pp.interp, include_params=include_interp_params)
                if include_tone_counts:
                    try:
                        n_tones = int(pp.start.freqs_hz.shape[0])
                    except Exception:
                        n_tones = -1
                    lc_descs.append(f"{lc}:{interp_desc}(tones={n_tones})")
                else:
                    lc_descs.append(f"{lc}:{interp_desc}")

            lines.append(f"  part {i}: {part_desc} " + " ".join(lc_descs))

        if hidden_parts:
            lines.append(f"  …(+{hidden_parts} more parts)")

    if hidden_segs:
        lines.append(f"- …(+{hidden_segs} more segments)")
    return "\n".join(lines)


def format_quantized_ir(
    q: QuantizedIR,
    *,
    unit: str = "us",
    max_segments: Optional[int] = None,
) -> str:
    """Format a `QuantizedIR` as a readable multi-line summary."""
    fs = float(q.sample_rate_hz)
    lines: list[str] = []
    lines.append(
        "QuantizedIR: "
        f"segments={len(q.segments)} "
        f"fs_hz={fs:.6g} "
        f"logical_channels={q.logical_channels}"
    )

    qis, hidden = _limit(q.quantization, max_items=max_segments)
    for qi in qis:
        lines.append(
            f"- segment {qi.name!r}: "
            f"{format_samples_time(qi.original_samples, fs, unit=unit)} -> "
            f"{format_samples_time(qi.quantized_samples, fs, unit=unit)} "
            f"loopable={qi.loopable} "
            f"step_samples={qi.step_samples} quantum_samples={qi.quantum_samples} "
            f"snap_len_to_quantum={qi.snap_len_to_quantum} snap_freqs_to_wrap={qi.snap_freqs_to_wrap}"
        )

    if hidden:
        lines.append(f"- …(+{hidden} more segments)")
    return "\n".join(lines)


def format_ir(
    ir: IntentIR | ResolvedIR | QuantizedIR | AWGProgramBuilder, **kwargs
) -> str:
    """Format any supported IR object (`IntentIR`, `ResolvedIR`, `QuantizedIR`, builder)."""
    if isinstance(ir, AWGProgramBuilder):
        return format_intent_ir(ir.build_intent_ir(), **kwargs)
    if isinstance(ir, IntentIR):
        return format_intent_ir(ir, **kwargs)
    if isinstance(ir, ResolvedIR):
        return format_resolved_ir(ir, **kwargs)
    if isinstance(ir, QuantizedIR):
        return format_quantized_ir(ir, **kwargs)
    raise TypeError(f"Unsupported IR type: {type(ir).__name__}")
