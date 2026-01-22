from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .ir import InterpKind
from .interpolation import interp_param


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

        x = t - self.t0
        freqs = interp_param(f0, f1, kind=self.interp, u=u, t_s=x, tau_s=self.tau_s)
        amps = interp_param(a0, a1, kind=self.interp, u=u, t_s=x, tau_s=self.tau_s)
        phases = interp_param(p0, p1, kind=self.interp, u=u, t_s=x, tau_s=self.tau_s)
        return LogicalChannelState(freqs, amps, phases)


@dataclass
class ResolvedTimeline:
    """
    Debug view of a resolved program as per-logical-channel time spans.

    This is intended for plotting / inspection (e.g. `state_at(...)`). The "real"
    compiler input is `ResolvedIR`.
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
