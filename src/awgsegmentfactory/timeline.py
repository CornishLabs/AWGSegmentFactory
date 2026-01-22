from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import numpy as np

InterpKind = Literal["hold", "linear", "exp", "min_jerk"]


@dataclass(frozen=True)
class LogicalChannelState:
    freqs_hz: np.ndarray  # (N,)
    amps: np.ndarray  # (N,)
    phases_rad: np.ndarray  # (N,)


@dataclass(frozen=True)
class Span:
    t0: float
    t1: float
    start: LogicalChannelState
    end: LogicalChannelState
    interp: InterpKind
    tau_s: Optional[float] = None
    seg_name: Optional[str] = None

    def duration(self) -> float:
        return self.t1 - self.t0

    def _smoothstep_minjerk(self, u: float) -> float:
        # classic 5th order minimum-jerk polynomial
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    def state_at(self, t: float) -> LogicalChannelState:
        if t <= self.t0:
            return self.start
        if t >= self.t1:
            return self.end
        dt = self.t1 - self.t0
        if dt <= 0:
            return self.end
        u = (t - self.t0) / dt

        f0, f1 = self.start.freqs_hz, self.end.freqs_hz
        a0, a1 = self.start.amps, self.end.amps
        p0, p1 = self.start.phases_rad, self.end.phases_rad

        if self.interp == "hold":
            return self.start

        if self.interp == "min_jerk":
            uu = self._smoothstep_minjerk(float(u))
            freqs = f0 + (f1 - f0) * uu
            amps = a0 + (a1 - a0) * uu
            phases = p0 + (p1 - p0) * uu
            return LogicalChannelState(freqs, amps, phases)

        if self.interp == "linear":
            freqs = f0 + (f1 - f0) * u
            amps = a0 + (a1 - a0) * u
            phases = p0 + (p1 - p0) * u
            return LogicalChannelState(freqs, amps, phases)

        if self.interp == "exp":
            # interpret as exponential approach for AMPLITUDES primarily; freqs/phases usually constant
            if self.tau_s is None or self.tau_s <= 0:
                freqs = f0 + (f1 - f0) * u
                amps = a0 + (a1 - a0) * u
                phases = p0 + (p1 - p0) * u
                return LogicalChannelState(freqs, amps, phases)
            x = t - self.t0
            k = np.exp(-x / self.tau_s)
            freqs = f1 + (f0 - f1) * k
            amps = a1 + (a0 - a1) * k
            phases = p1 + (p0 - p1) * k
            return LogicalChannelState(freqs, amps, phases)

        raise ValueError(f"Unknown interp {self.interp!r}")


@dataclass
class ResolvedTimeline:
    """
    Debug view of a resolved program as per-logical-channel time spans.

    This is intended for plotting / inspection (e.g. `state_at(...)`). The "real"
    compiler input is `ProgramIR`.
    """

    sample_rate_hz: float
    logical_channels: Dict[str, List[Span]]
    segment_starts: List[tuple[float, str]]
    t_end: float

    def state_at(self, logical_channel: str, t: float) -> LogicalChannelState:
        spans = self.logical_channels[logical_channel]
        if not spans:
            raise ValueError(f"No spans available for logical_channel {logical_channel!r}")
        if t <= spans[0].t0:
            return spans[0].start
        prev_end = spans[0].end
        for sp in spans:
            if sp.t0 <= t <= sp.t1:
                return sp.state_at(t)
            if t < sp.t0:
                # Gap: logical channel holds its last value until the next span begins.
                return prev_end
            prev_end = sp.end
        return spans[-1].end

    def sample_times(self, fps: float = 200.0) -> np.ndarray:
        n = max(2, int(self.t_end * fps) + 1)
        return np.linspace(0.0, self.t_end, n)
