from .builder import AWGProgramBuilder
from .calibration import LinearPositionToFreqCalib
from .program_ir import ProgramIR

__all__ = [
    "AWGProgramBuilder",
    "LinearPositionToFreqCalib",
    "ProgramIR",
    "interactive_grid_debug",
    "LinearFreqToPos",
]


def __getattr__(name: str):
    if name in {"interactive_grid_debug", "LinearFreqToPos"}:
        try:
            from .debug_plot import interactive_grid_debug, LinearFreqToPos
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                f"`awgsegmentfactory.{name}` requires the optional debug-plot dependencies. "
                "Install the `dev` dependency group (matplotlib, ipywidgets, etc.)."
            ) from exc
        return interactive_grid_debug if name == "interactive_grid_debug" else LinearFreqToPos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
