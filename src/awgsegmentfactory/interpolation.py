"""Interpolation primitives shared by timeline debug and sample synthesis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from .intent_ir import InterpSpec


def _dtype_like(x: Any, *, default: Any = float) -> Any:
    return getattr(x, "dtype", default)


def smoothstep_min_jerk(u: float | np.ndarray) -> float | np.ndarray:
    """5th-order minimum-jerk smoothstep mapping `u` in [0, 1] -> [0, 1]."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def adiabatic_ramp_powerlaw(
    start: Any,
    end: Any,
    u: Any,
    a: float,
    smooth: bool = False,
    eps: float = 0.0,
    *,
    xp: Any = np,
):
    """
    Constant-adiabaticity ramp when trap frequency scales as ω ∝ x^a.

    a = 1   -> ω ∝ x        (typical if x is AWG amplitude and I ∝ x^2)
    a = 1/2 -> ω ∝ sqrt(x)  (typical if x is optical power / intensity)
    """
    dtype = _dtype_like(start)
    u = xp.asarray(u, dtype=dtype)
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


def adiabatic_ramp_awg_amp(
    start: Any, end: Any, u: Any, smooth: bool = False, eps: float = 0.0, *, xp: Any = np
):
    """Convenience wrapper for `adiabatic_ramp_powerlaw(..., a=1.0)`."""
    return adiabatic_ramp_powerlaw(
        start, end, u, a=1.0, smooth=smooth, eps=eps, xp=xp
    )


def _interp_hold(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    return start


def _interp_linear(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    return start + (end - start) * u


def _interp_min_jerk(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    dtype = _dtype_like(start)
    uu = smoothstep_min_jerk(xp.asarray(u, dtype=dtype))
    return start + (end - start) * uu


def _interp_exp(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    if t_s is None:
        raise ValueError("exp interpolation requires t_s")
    if interp.tau_s is None:  # pragma: no cover
        raise ValueError("exp interpolation requires interp.tau_s")

    dtype = _dtype_like(start)
    t = xp.asarray(t_s, dtype=dtype)
    tau = abs(float(interp.tau_s))
    if tau > 0.0:
        k = xp.exp(-t / tau)
    else:
        # tau -> 0+: start at t=0, otherwise jump to end immediately
        k = xp.where(t == 0.0, 1.0, 0.0)
    return end + (start - end) * k


def _interp_geo_ramp(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    dtype = _dtype_like(start)
    uu = xp.asarray(u, dtype=dtype)

    def _any(x: Any) -> bool:
        return bool(xp.any(x).item())

    if _any(start == 0) or _any(end == 0) or _any(xp.sign(start) != xp.sign(end)):
        raise ValueError("geo_ramp requires start and end to be nonzero and same sign")
    return start * xp.exp(xp.log(end / start) * uu)


def _interp_adiabatic_ramp(
    start: Any,
    end: Any,
    *,
    interp: InterpSpec,
    u: Any,
    t_s: Any | None,
    xp: Any,
) -> Any:
    dtype = _dtype_like(start)
    uu = xp.asarray(u, dtype=dtype)
    return adiabatic_ramp_awg_amp(start, end, uu, smooth=True, eps=0.005, xp=xp)


_INTERP_BY_KIND: dict[str, Callable[..., Any]] = {
    "hold": _interp_hold,
    "linear": _interp_linear,
    "min_jerk": _interp_min_jerk,
    "exp": _interp_exp,
    "geo_ramp": _interp_geo_ramp,
    "adiabatic_ramp": _interp_adiabatic_ramp,
}


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

    - For `linear`/`min_jerk`/`geo_ramp`/`adiabatic_ramp`, interpolation is controlled by `u` in [0, 1).
    - For `exp`, interpolation is controlled by elapsed time `t_s` (seconds) and `interp.tau_s`.

    Shapes:
    - `start` and `end` must have the same shape.
    - `u`/`t_s` may be scalars or arrays and will broadcast against `start`.
    """
    if start.shape != end.shape:
        raise ValueError("Start/end shape mismatch")

    kind = str(interp.kind)
    try:
        fn = _INTERP_BY_KIND[kind]
    except KeyError as exc:
        known = ", ".join(repr(k) for k in sorted(_INTERP_BY_KIND.keys()))
        raise ValueError(f"Unknown interpolation kind {kind!r}; expected one of: {known}") from exc

    return fn(start, end, interp=interp, u=u, t_s=t_s, xp=xp)
