"""Interpolation primitives shared by timeline debug and sample synthesis."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .intent_ir import InterpSpec


def smoothstep_min_jerk(u: float | np.ndarray) -> float | np.ndarray:
    """5th-order minimum-jerk smoothstep mapping `u` in [0, 1] -> [0, 1]."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def adiabatic_ramp_powerlaw(start, end, u, a, smooth=False, eps=0.0, *, xp: Any = np):
    """
    Constant-adiabaticity ramp when trap frequency scales as ω ∝ x^a.

    a = 1   -> ω ∝ x        (typical if x is AWG amplitude and I ∝ x^2)
    a = 1/2 -> ω ∝ sqrt(x)  (typical if x is optical power / intensity)
    """
    u = xp.asarray(u, dtype=float)
    if smooth:
        u = u * u * (3.0 - 2.0 * u)  # smoothstep

    # Avoid singularities if you ever ramp near zero

    x0 = xp.maximum(start, eps)
    x1 = xp.maximum(end, eps)

    q0 = x0 ** (-a)
    q1 = x1 ** (-a)

    # Broadcast u over the tone axis.
    # `u` often arrives as (n_samples, 1); don't add an extra singleton axis or we'd
    # return (n_samples, 1, n_tones) which breaks downstream synthesis.
    if u.ndim == 1:
        u = u[:, None]

    q = (1.0 - u) * q0 + u * q1
    return q ** (-1.0 / a)


def adiabatic_ramp_awg_amp(start, end, u, smooth=False, eps=0.0, *, xp: Any = np):
    return adiabatic_ramp_powerlaw(
        start, end, u, a=1.0, smooth=smooth, eps=eps, xp=xp
    )


def interp_param(
    start: np.ndarray,
    end: np.ndarray,
    *,
    interp: InterpSpec,
    u: float | np.ndarray,
    t_s: Optional[float | np.ndarray] = None,
    xp: Any = np,
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

        if float(interp.tau_s) > 0:
            k = xp.exp(-xp.asarray(t_s, dtype=float) / float(interp.tau_s))
            return end + (start - end) * k
        else:
            k = xp.exp(-xp.asarray(t_s, dtype=float) / float(-interp.tau_s))
            return end + (start - end) * k
        
    if kind == "geo_ramp":
        u = xp.asarray(u, dtype=float)

        if (
            xp.any(start == 0)
            or xp.any(end == 0)
            or xp.any(xp.sign(start) != xp.sign(end))
        ):
            raise ValueError("geo_ramp requires start and end to be nonzero and same sign")
        return start * xp.exp(xp.log(end / start) * u)
    
    if kind == "adiabatic_ramp":
        u = xp.asarray(u, dtype=float)
        return adiabatic_ramp_awg_amp(start, end, u, smooth=True, eps=0.02, xp=xp)

    uu: float | np.ndarray
    if kind == "min_jerk":
        uu = smoothstep_min_jerk(xp.asarray(u, dtype=float))
    else:
        uu = u

    return start + (end - start) * uu
