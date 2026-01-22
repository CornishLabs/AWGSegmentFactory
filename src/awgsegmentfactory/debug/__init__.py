from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "interactive_grid_debug",
    "LinearFreqToPos",
    "sequence_samples_debug",
    "PipelineTimings",
    "benchmark_builder_pipeline",
    "compile_builder_pipeline_timed",
    "format_benchmark_table",
]

if TYPE_CHECKING:  # pragma: no cover
    from .plot import LinearFreqToPos, interactive_grid_debug
    from .perf import (
        PipelineTimings,
        benchmark_builder_pipeline,
        compile_builder_pipeline_timed,
        format_benchmark_table,
    )
    from .samples import sequence_samples_debug


def __getattr__(name: str):
    if name in {"interactive_grid_debug", "LinearFreqToPos"}:
        from .plot import LinearFreqToPos, interactive_grid_debug

        return (
            interactive_grid_debug
            if name == "interactive_grid_debug"
            else LinearFreqToPos
        )
    if name == "sequence_samples_debug":
        from .samples import sequence_samples_debug

        return sequence_samples_debug
    if name in {
        "PipelineTimings",
        "benchmark_builder_pipeline",
        "compile_builder_pipeline_timed",
        "format_benchmark_table",
    }:
        from .perf import (
            PipelineTimings,
            benchmark_builder_pipeline,
            compile_builder_pipeline_timed,
            format_benchmark_table,
        )

        return {
            "PipelineTimings": PipelineTimings,
            "benchmark_builder_pipeline": benchmark_builder_pipeline,
            "compile_builder_pipeline_timed": compile_builder_pipeline_timed,
            "format_benchmark_table": format_benchmark_table,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
