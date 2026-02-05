"""Optional debug/inspection helpers.

These helpers are intentionally separated from the core compiler pipeline and are
imported lazily so `awgsegmentfactory` can be used without matplotlib/ipywidgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "interactive_grid_debug",
    "LinearFreqToPos",
    "sequence_samples_debug",
    "format_intent_ir",
    "format_resolved_ir",
    "format_quantized_ir",
    "format_ir",
    "plot_sin2_fit_report",
    "plot_sin2_fit_report_by_logical_channel",
    "PipelineTimings",
    "benchmark_builder_pipeline",
    "compile_builder_pipeline_timed",
    "format_benchmark_table",
]

if TYPE_CHECKING:  # pragma: no cover
    from .plot import LinearFreqToPos, interactive_grid_debug
    from .ir import format_intent_ir, format_resolved_ir, format_quantized_ir, format_ir
    from .optical_power_calibration import (
        plot_sin2_fit_report,
        plot_sin2_fit_report_by_logical_channel,
    )
    from .perf import (
        PipelineTimings,
        benchmark_builder_pipeline,
        compile_builder_pipeline_timed,
        format_benchmark_table,
    )
    from .samples import sequence_samples_debug


def __getattr__(name: str):
    """Lazy-load debug helpers to avoid importing heavy optional dependencies."""
    if name in {"format_intent_ir", "format_resolved_ir", "format_quantized_ir", "format_ir"}:
        from .ir import format_intent_ir, format_ir, format_quantized_ir, format_resolved_ir

        return {
            "format_intent_ir": format_intent_ir,
            "format_resolved_ir": format_resolved_ir,
            "format_quantized_ir": format_quantized_ir,
            "format_ir": format_ir,
        }[name]
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
    if name in {"plot_sin2_fit_report", "plot_sin2_fit_report_by_logical_channel"}:
        from .optical_power_calibration import (
            plot_sin2_fit_report,
            plot_sin2_fit_report_by_logical_channel,
        )

        return {
            "plot_sin2_fit_report": plot_sin2_fit_report,
            "plot_sin2_fit_report_by_logical_channel": plot_sin2_fit_report_by_logical_channel,
        }[name]
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
