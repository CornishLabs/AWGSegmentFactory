"""Interpolation primitives shared by timeline debug and sample synthesis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ir import InterpKind


def smoothstep_min_jerk(u: float | np.ndarray) -> float | np.ndarray:
    """5th-order minimum-jerk smoothstep mapping `u` in [0, 1] -> [0, 1]."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def interp_param(
    start: np.ndarray,
    end: np.ndarray,
    *,
    kind: InterpKind,
    u: float | np.ndarray,
    t_s: Optional[float | np.ndarray] = None,
    tau_s: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate a parameter array from `start` -> `end`.

    - For `linear`/`min_jerk`, interpolation is controlled by `u` in [0, 1).
    - For `exp`, interpolation is controlled by elapsed time `t_s` (seconds) and `tau_s`.
      If `tau_s` is missing/invalid, falls back to linear interpolation using `u`.

    Shapes:
    - `start` and `end` must have the same shape.
    - `u`/`t_s` may be scalars or arrays and will broadcast against `start`.
    """
    if start.shape != end.shape:
        raise ValueError("Start/end shape mismatch")

    if kind == "hold":
        return start

    if kind == "exp":
        if t_s is None or tau_s is None or tau_s <= 0:
            return start + (end - start) * u
        k = np.exp(-np.asarray(t_s, dtype=float) / float(tau_s))
        return end + (start - end) * k

    uu: float | np.ndarray
    if kind == "min_jerk":
        uu = smoothstep_min_jerk(np.asarray(u, dtype=float))
    else:
        uu = u

    return start + (end - start) * uu
