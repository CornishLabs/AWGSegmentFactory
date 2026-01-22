"""
Show the different "depths" of compilation for AWGSegmentFactory:

1) Builder (high-level user intent)
2) ProgramSpec (intent IR / recorded ops)
3) ProgramIR (segment-grouped, sample-quantised primitives)
4) ResolvedTimeline (debug view for plotting / state queries)

This is also a sketch of how a NumPy/CuPy backend could turn `ProgramIR`
parts into vectorised arrays (freq/amp/phase per sample).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import numpy as np
import warnings

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.program_ir import PlanePartIR


def _describe_spec(spec) -> None:
    print("\n== ProgramSpec (intent IR) ==")
    print(f"sample_rate_hz: {spec.sample_rate_hz}")
    print(f"planes: {spec.planes}")
    print(f"definitions: {list(spec.definitions.keys())}")
    for seg in spec.segments:
        op_names = [type(op).__name__ for op in seg.ops]
        print(f"- segment {seg.name!r}: mode={seg.mode} loop={seg.loop} ops={op_names}")


def _describe_ir(ir) -> None:
    print("\n== ProgramIR (primitives) ==")
    print(
        f"segments: {len(ir.segments)} | total_samples: {ir.n_samples} | duration_s: {ir.duration_s:.6f}"
    )
    for seg in ir.segments:
        print(
            f"- segment {seg.name!r}: mode={seg.mode} loop={seg.loop} samples={seg.n_samples}"
        )
        for i, part in enumerate(seg.parts):
            kinds = {p: part.planes[p].interp for p in ir.planes}
            print(f"  part {i}: n_samples={part.n_samples} kinds={kinds}")


def _interp_min_jerk(u: np.ndarray) -> np.ndarray:
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _plane_part_to_arrays(
    pp: PlanePartIR,
    *,
    n_samples: int,
    sample_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorise a PlanePartIR into per-sample arrays (freqs/amps/phases).

    Notes:
    - This is a *demonstration*; the final AWG compiler will want to define a precise
      convention for endpoint handling (sample centers vs edges).
    - For a true chirp waveform, you should integrate instantaneous frequency into phase
      (see `_tonebank_to_samples`).
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    # sample times relative to the part start
    t = np.arange(n_samples, dtype=float) / float(sample_rate_hz)

    f0, f1 = pp.start.freqs_hz, pp.end.freqs_hz
    a0, a1 = pp.start.amps, pp.end.amps
    p0, p1 = pp.start.phases_rad, pp.end.phases_rad

    if pp.interp == "hold":
        freqs = np.repeat(f0[None, :], n_samples, axis=0)
        amps = np.repeat(a0[None, :], n_samples, axis=0)
        phases = np.repeat(p0[None, :], n_samples, axis=0)
        return freqs, amps, phases

    # u in [0,1)
    u = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=float)[:, None]

    if pp.interp == "min_jerk":
        u = _interp_min_jerk(u)

    if pp.interp == "linear" or pp.interp == "min_jerk":
        freqs = f0[None, :] + (f1 - f0)[None, :] * u
        amps = a0[None, :] + (a1 - a0)[None, :] * u
        phases = p0[None, :] + (p1 - p0)[None, :] * u
        return freqs, amps, phases

    if pp.interp == "exp":
        if pp.tau_s is None or pp.tau_s <= 0:
            freqs = f0[None, :] + (f1 - f0)[None, :] * u
            amps = a0[None, :] + (a1 - a0)[None, :] * u
            phases = p0[None, :] + (p1 - p0)[None, :] * u
            return freqs, amps, phases
        k = np.exp(-t[:, None] / float(pp.tau_s))
        freqs = f1[None, :] + (f0 - f1)[None, :] * k
        amps = a1[None, :] + (a0 - a1)[None, :] * k
        phases = p1[None, :] + (p0 - p1)[None, :] * k
        return freqs, amps, phases

    raise ValueError(f"Unknown interp {pp.interp!r}")


def _tonebank_to_samples(
    freqs_hz: np.ndarray,
    amps: np.ndarray,
    phases0_rad: np.ndarray,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    Demonstration: true chirp synthesis by phase integration.

    - freqs_hz: (N, T) instantaneous frequency per tone
    - amps:     (N, T) amplitude per tone
    - phases0_rad: (T,) starting phase per tone
    Returns:
    - y: (N,) summed waveform
    """

    dt = 1.0 / float(sample_rate_hz)
    dphi = 2.0 * np.pi * freqs_hz * dt
    phi = phases0_rad[None, :] + np.cumsum(dphi, axis=0)
    return np.sum(amps * np.sin(phi), axis=1)


def main() -> None:
    fs = 10.0  # small so sample rounding is obvious

    # ---- 1) Builder (high-level intent) ----
    b = AWGProgramBuilder(sample_rate=fs).plane("H").plane("V")
    b.define("init_H", plane="H", freqs=[7.0, 9.0], amps=[1.0, 0.5], phases="auto")
    b.define("init_V", plane="V", freqs=[100.0], amps=[1.0], phases="auto")

    b.segment("sync", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=0.25, warn_df=0.01)  # 0.25s -> ceil(2.5)=3 samples -> dt=0.3s snap grid

    b.segment("move_H", mode="once")
    b.tones("H").move(df=+1.0, time=0.3)  # 0.3s -> 3 samples

    b.segment("ramp_V", mode="loop_n", loop=2)
    b.tones("V").ramp_amp_to(
        amps=0.0, time=0.2, kind="exp", tau=0.1
    )  # 0.2s -> 2 samples

    # ---- 2) ProgramSpec (intent IR) ----
    spec = b.build_spec()
    _describe_spec(spec)

    # ---- 3) ProgramIR (segment-grouped primitives) ----
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ir = b.build_ir()
    _describe_ir(ir)
    if w:
        for ww in w:
            print(f"[warning] {ww.message}")

    # ---- 4) ResolvedTimeline (debug view) ----
    tl = ir.to_timeline()
    print("\n== ResolvedTimeline (debug view) ==")
    print(f"t_end: {tl.t_end:.6f}s | segment_starts: {tl.segment_starts}")
    print("state_at(H, t=0.05):", asdict(tl.state_at("H", 0.05)))

    # ---- 5) Example: vectorise a single segment+plane into samples ----
    seg0 = ir.segments[0]
    part0 = seg0.parts[0]
    ppH = part0.planes["H"]
    freqs, amps, phases = _plane_part_to_arrays(
        ppH, n_samples=part0.n_samples, sample_rate_hz=ir.sample_rate_hz
    )
    y = _tonebank_to_samples(
        freqs, amps, ppH.start.phases_rad, sample_rate_hz=ir.sample_rate_hz
    )
    print("\n== Vectorised synthesis demo ==")
    print(
        f"segment {seg0.name!r} plane 'H': freqs shape={freqs.shape} amps shape={amps.shape} y shape={y.shape}"
    )
    print("first few y samples:", np.array2string(y[: min(len(y), 8)], precision=4))


if __name__ == "__main__":
    main()
