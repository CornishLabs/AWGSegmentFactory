"""Interpolation primitives shared by timeline debug and sample synthesis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .intent_ir import InterpSpec


def smoothstep_min_jerk(u: float | np.ndarray) -> float | np.ndarray:
    """5th-order minimum-jerk smoothstep mapping `u` in [0, 1] -> [0, 1]."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def interp_param(
    start: np.ndarray,
    end: np.ndarray,
    *,
    interp: InterpSpec,
    u: float | np.ndarray,
    t_s: Optional[float | np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate a parameter array from `start` -> `end`.

    - For `linear`/`min_jerk`, interpolation is controlled by `u` in [0, 1).
    - For `exp`, interpolation is controlled by elapsed time `t_s` (seconds) and `interp.tau_s`.

    Shapes:
    - `start` and `end` must have the same shape.
    - `u`/`t_s` may be scalars or arrays and will broadcast against `start`.
    """
    if start.shape != end.shape:
        raise ValueError("Start/end shape mismatch")

    kind = interp.kind
    if kind == "hold":
        return start

    if kind == "exp":
        if t_s is None:
            raise ValueError("exp interpolation requires t_s")
        if interp.tau_s is None:  # pragma: no cover
            raise ValueError("exp interpolation requires interp.tau_s")
        k = np.exp(-np.asarray(t_s, dtype=float) / float(interp.tau_s))
        return end + (start - end) * k

    uu: float | np.ndarray
    if kind == "min_jerk":
        uu = smoothstep_min_jerk(np.asarray(u, dtype=float))
    else:
        uu = u

    return start + (end - start) * uu
