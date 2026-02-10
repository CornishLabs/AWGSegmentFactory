"""Performance helpers for timing the builder → samples compilation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Callable, Optional, Sequence

from ..builder import AWGProgramBuilder
from ..calibration import AWGPhysicalSetupInfo
from ..intent_ir import IntentIR
from ..resolve import resolve_intent_ir
from ..synth_samples import CompiledSequenceProgram, compile_sequence_program
from ..quantize import quantize_resolved_ir


@dataclass(frozen=True)
class PipelineTimings:
    """Wall-clock timings for each stage of the compilation pipeline."""

    build_intent_s: float
    intent_codec_s: float
    resolve_s: float
    quantize_s: float
    compile_s: float
    total_s: float


def compile_builder_pipeline_timed(
    builder: AWGProgramBuilder,
    *,
    sample_rate_hz: float,
    physical_setup: Optional[AWGPhysicalSetupInfo] = None,
    gain: float = 1.0,
    clip: float = 0.9,
    full_scale: int = 32767,
    gpu: bool = False,
) -> tuple[CompiledSequenceProgram, PipelineTimings]:
    """
    Compile end-to-end (Builder -> IntentIR -> ResolvedIR -> QuantizedIR -> samples),
    returning both the compiled program and per-stage wall-clock timings.

    If `physical_setup` is omitted, an identity logical->hardware mapping is used.
    """
    t0 = perf_counter()
    intent = builder.build_intent_ir()
    t1 = perf_counter()

    # Simulate shipping IntentIR "over the wire": encode -> hash -> decode.
    encoded_intent = intent.encode()
    hashlib.sha256(
        json.dumps(
            encoded_intent, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    intent = IntentIR.decode(encoded_intent)
    t2 = perf_counter()

    resolved = resolve_intent_ir(intent, sample_rate_hz=sample_rate_hz)
    t3 = perf_counter()
    quantized = quantize_resolved_ir(resolved)
    t4 = perf_counter()
    setup = physical_setup
    if setup is None:
        setup = AWGPhysicalSetupInfo.identity(quantized.logical_channels)
    compiled = compile_sequence_program(
        quantized,
        physical_setup=setup,
        gain=gain,
        clip=clip,
        full_scale=full_scale,
        gpu=gpu,
    )
    if gpu:
        # CuPy is asynchronous; ensure timings include queued GPU work.
        try:  # pragma: no cover
            import cupy as cp  # type: ignore

            cp.cuda.Stream.null.synchronize()
        except Exception:
            # If CuPy isn't available here, compile_sequence_program(gpu=True) should
            # already have failed; keep this as a best-effort sync.
            pass
    t5 = perf_counter()

    return compiled, PipelineTimings(
        build_intent_s=t1 - t0,
        intent_codec_s=t2 - t1,
        resolve_s=t3 - t2,
        quantize_s=t4 - t3,
        compile_s=t5 - t4,
        total_s=t5 - t0,
    )


def benchmark_builder_pipeline(
    build_builder: Callable[[], AWGProgramBuilder],
    *,
    sample_rate_hz: float,
    physical_setup: Optional[AWGPhysicalSetupInfo] = None,
    iters: int = 5,
    warmup: int = 1,
    gain: float = 1.0,
    clip: float = 0.9,
    full_scale: int = 32767,
    gpu: bool = False,
) -> tuple[PipelineTimings, ...]:
    """
    Run repeated end-to-end compiles, returning timings per iteration.

    Notes:
    - Uses a fresh builder per iteration (`build_builder`) so you benchmark the
      real "user code -> samples" path.
    - Uses wall-clock timing (`perf_counter`).
    """
    if iters <= 0:
        raise ValueError("iters must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    runs: list[PipelineTimings] = []
    for i in range(warmup + iters):
        builder = build_builder()
        _compiled, timing = compile_builder_pipeline_timed(
            builder,
            sample_rate_hz=sample_rate_hz,
            physical_setup=physical_setup,
            gain=gain,
            clip=clip,
            full_scale=full_scale,
            gpu=gpu,
        )
        if i >= warmup:
            runs.append(timing)
    return tuple(runs)


def _summarize(values_s: Sequence[float]) -> tuple[float, float, float]:
    """Return (min, mean, max) for a list of timings in seconds."""
    if not values_s:
        raise ValueError("values_s must be non-empty")
    mn = min(values_s)
    mx = max(values_s)
    mean = sum(values_s) / len(values_s)
    return mn, mean, mx


def format_benchmark_table(runs: Sequence[PipelineTimings]) -> str:
    """
    Format a simple text table summarizing benchmark results (min/mean/max).
    """
    if not runs:
        raise ValueError("runs must be non-empty")

    def col(get) -> tuple[float, float, float]:
        """Summarize one timing field across runs as (min, mean, max)."""
        return _summarize([float(get(r)) for r in runs])

    rows = [
        ("build_intent", col(lambda r: r.build_intent_s)),
        ("intent_codec", col(lambda r: r.intent_codec_s)),
        ("resolve", col(lambda r: r.resolve_s)),
        ("quantize", col(lambda r: r.quantize_s)),
        ("compile", col(lambda r: r.compile_s)),
        ("total", col(lambda r: r.total_s)),
    ]

    def fmt_s(x: float) -> str:
        """Format seconds for the table (ms for sub-second values)."""
        if x >= 1.0:
            return f"{x:8.3f}s"
        return f"{x*1e3:8.3f}ms"

    out = ["stage         min        mean        max"]
    for name, (mn, mean, mx) in rows:
        out.append(f"{name:<12} {fmt_s(mn)} {fmt_s(mean)} {fmt_s(mx)}")
    return "\n".join(out)
