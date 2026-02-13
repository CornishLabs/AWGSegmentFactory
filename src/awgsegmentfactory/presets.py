"""Reusable builder/Intent presets used by examples and tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .builder import AWGProgramBuilder
import numpy as np

if TYPE_CHECKING:
    from .intent_ir import IntentIR


def ramp_down_chirp_2ch() -> AWGProgramBuilder:
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define("init_H", logical_channel="H", freqs=[1e5], amps=[0.3], phases="auto")
        .define("init_V", logical_channel="V", freqs=[2e5], amps=[0.3], phases="auto")
    )

    # 0) Wait for trigger, output stable tones (this segment loops until a trigger event)
    b.segment("wait_start", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=40e-6)

    b.segment("ramp_down", mode="once")
    b.tones("V").ramp_amp_to(
        amps=[0.1],
        time=50e-6,
        kind="exp",
        tau=20e-6,
    )

    # 1) A short chirp on H (one-shot segment)
    b.segment("chirp_H", mode="once")
    b.tones("H").move(df=+1e5, time=100e-6, idxs=[0], kind="linear")


    b.segment("wait_start_2", mode="wait_trig")
    b.hold(time=40e-6)

    return b


def spec_analyser_test() -> AWGProgramBuilder:
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define("init_H", logical_channel="H", freqs=[1e5], amps=[0.5], phases="auto")
        .define("init_V", logical_channel="V", freqs=[2e4], amps=[0.5], phases="auto")
    )

    # 0) Wait for trigger, output stable tones (this segment loops until a trigger event)
    b.segment("wait_start", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=40e-6)

    b.segment("move", mode="once")
    b.tones("H").move(df=+1e5, time=100e-3, idxs=[0], kind="linear")

    b.segment("wait_start_2", mode="wait_trig")
    b.hold(time=40e-6)

    return b

def rt_spec_analyser_rearr_hotswap() -> AWGProgramBuilder:
    """
    The low frequencies here are for use with a realtime spectrum analyser at
    a lower sample rate
    uv run aqctl_spectrum_awg \
    --serial-number 14926 -vv --bind ::1 \
    --characterisation-lookup-str AWG_2CH_MV --sample-rate 62500000 --gpu
    """
    b=(
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define(
            "loading_H",
            logical_channel="H",
            freqs=np.linspace(20e3, 150e3, 10),
            amps=[50] * 10,
            phases="auto",
        )
        .define("loading_V", logical_channel="V", freqs=[0], amps=[0], phases="auto")
        .define(
            "exp_H",
            logical_channel="H",
            freqs=np.linspace(30e3, 100e3, 4),
            amps=[50] * 4,
            phases="auto",
        )
        .define("exp_V", logical_channel="V", freqs=[100e6], amps=[0.7], phases="auto")
        # 1) Initial sync: minimum length, wait for first trigger
        .segment("sync", mode="wait_trig", snap_len_to_quantum=False)
            .hold(
                time=1e-9
            )  # wait_trig defaults: wrap-snap freqs; snap_len_to_quantum=False keeps trigger latency minimal
        # 2) Loading tweezers on: wait for trigger then output steady tones
        .segment("loading_tweezers_on", mode="wait_trig", phase_mode="optimise")
            .tones("H")
            .use_def("loading_H")
            .tones("V")
            .use_def("loading_V")
            .hold(
                time=400e-6
            )
        # 3) Rearrange (hotswappable): only H changes (V implicitly unchanged)
        #
        # This is *not* a “move(df=...)” — it’s a remapping/re-targeting operation:
        # “Take selected existing tones, map them onto the target definition ordering,
        #  dropping extras, and tween over time with a chosen curve.”
        .segment("hotswap_rearrange_to_exp_array", mode="loop_n", loop=1, phase_mode="continue")
            .tones("H")
            .remap_from_def(
                target_def="exp_H",
                src=[2, 4, 8, 9],
                dst="all",
                time=2.5e-3, 
                kind="min_jerk",
            )
        # 5) Wait for trigger: wrap-continuous quantised hold
        .segment("wait_for_trigger_A", mode="wait_trig")
            .hold(time=40e-6)
    )

    return b


def simple_tweens() -> AWGProgramBuilder:
    fs = 1e9
    one_sample = 1.0 / fs

    # Two traps, separated along H, same initial V position
    H0, H1 = 90e6, 110e6
    V0 = 100e6

    prog = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        # Define initial trap states (paired indexing: trap i = (H[i], V[i]))
        .define("start_H", logical_channel="H", freqs=[H0, H1], amps=[0.7, 0.7], phases="auto")
        .define("start_V", logical_channel="V", freqs=[V0, V0], amps=[0.7, 0.7], phases="auto")
        # 0) arm / sync
        .segment("sync", mode="wait_trig")
            .tones("H")
            .use_def("start_H")
            .tones("V")
            .use_def("start_V")
            .hold(time=200e-6)  # wait; compiler will quantise freqs to wrap this length
            # ---- Trap 0 (left) ----
        # 1) move left trap up (V only, idx 0)
        .segment("left_up", mode="once")
            .tones("V")
            .move(df=+2.0e6, time=200e-6, idxs=[0])
        # 2) move left trap right (H only, idx 0)
        .segment("left_right", mode="once")
            .tones("H")
            .move(df=+1.0e6, time=200e-6, idxs=[0])
        # 3) ramp it down (amp of trap 0 on both logical channels; keep trap “present”)
        .segment("left_ramp_down", mode="once")
        .tones("H")
        .ramp_amp_to(amps=[0.1], time=1e-3, idxs=[0], kind="exp", tau=0.2e-3)
        .tones("V")
        .ramp_amp_to(amps=[0.1], time=1e-3, idxs=[0], kind="exp", tau=0.2e-3)
        # 4) undo: ramp up, move left, move down (all within one segment)
        .segment("left_undo", mode="once")
        .tones("H")
        .ramp_amp_to(amps=[0.7], time=1e-3, idxs=[0], kind="exp", tau=0.2e-3)
        .tones("V")
        .ramp_amp_to(amps=[0.7], time=1e-3, idxs=[0], kind="exp", tau=0.2e-3)
        .tones("H")
        .move(df=-1.0e6, time=200e-6, idxs=[0])
        .tones("V")
        .move(df=-2.0e6, time=200e-6, idxs=[0])
        # (optional) wait-for-trigger “rest” segment, quantised wrapping
        .segment("wait_A", mode="wait_trig")
        .hold(time=200e-6)
        # ---- Trap 1 (right) ----
        .segment("right_up", mode="once")
        .tones("V")
        .move(df=+2.0e6, time=200e-6, idxs=[1])
        .segment("right_right", mode="once")
        .tones("H")
        .move(df=+1.0e6, time=200e-6, idxs=[1])
        .segment("right_ramp_down", mode="once")
        .tones("H")
        .ramp_amp_to(amps=[0.1], time=1e-3, idxs=[1], kind="exp", tau=0.2e-3)
        .tones("V")
        .ramp_amp_to(amps=[0.1], time=1e-3, idxs=[1], kind="exp", tau=0.2e-3)
        .segment("right_undo", mode="once")
        .tones("H")
        .ramp_amp_to(amps=[0.7], time=1e-3, idxs=[1], kind="exp", tau=0.2e-3)
        .tones("V")
        .ramp_amp_to(amps=[0.7], time=1e-3, idxs=[1], kind="exp", tau=0.2e-3)
        .tones("H")
        .move(df=-1.0e6, time=200e-6, idxs=[1])
        .tones("V")
        .move(df=-2.0e6, time=200e-6, idxs=[1])
        # final segment must be >= 1 sample too (even if you “do nothing”)
        .segment("done", mode="wait_trig")
        .hold(time=max(200e-6, one_sample))
    )

    return prog

def recreate_mol_exp() -> AWGProgramBuilder:
    ir = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        # .grid("twz", H="H", V="V")  # optional metadata only; not needed for control
        # Definitions (do not “play” yet)
        .define(
            "loading_H",
            logical_channel="H",
            freqs=np.linspace(92.0e6, 110.0e6, 12),
            amps=[0.08] * 12,
            phases="auto",
        )
        .define("loading_V", logical_channel="V", freqs=[100e6], amps=[0.7], phases="auto")
        .define(
            "exp_H",
            logical_channel="H",
            freqs=np.linspace(94.0e6, 108.0e6, 8),
            amps=[0.08] * 8,
            phases="auto",
        )
        .define("exp_V", logical_channel="V", freqs=[100e6], amps=[0.7], phases="auto")
        # 1) Initial sync: minimum length, wait for first trigger
        .segment("sync", mode="wait_trig", snap_len_to_quantum=False)
            .hold(
                time=1e-9
            )  # wait_trig defaults: wrap-snap freqs; snap_len_to_quantum=False keeps trigger latency minimal
        # 2) Loading tweezers on: wait for trigger then output steady tones
        .segment("loading_tweezers_on", mode="wait_trig", phase_mode="optimise")
            .tones("H")
            .use_def("loading_H")
            .tones("V")
            .use_def("loading_V")
            .hold(
                time=200e-6
            )
        # 3) Rearrange (hotswappable): only H changes (V implicitly unchanged)
        #
        # This is *not* a “move(df=...)” — it’s a remapping/re-targeting operation:
        # “Take selected existing tones, map them onto the target definition ordering,
        #  dropping extras, and tween over time with a chosen curve.”
        .segment("hotswap_rearrange_to_exp_array", mode="loop_n", loop=1, phase_mode="continue")
            .tones("H")
            .remap_from_def(
                target_def="exp_H",
                src=[2, 4, 5, 7, 8, 9, 10, 11],
                dst="all",
                time=1e-3,
                kind="min_jerk",
            )
        # 4) Equalise amps (H only)
        .segment("equalise_amps", mode="loop_n", loop=1, phase_mode="optimise")
            .tones("H")
            .ramp_amp_to(
                amps=0.15*np.array([0.73, 0.68, 0.67, 0.75, 0.62, 0.81, 0.74, 0.73]),
                time=1e-3,
                kind="adiabatic_ramp"
            )
        # 5) Wait for trigger: wrap-continuous quantised hold
        .segment("wait_for_trigger_A", mode="wait_trig")
            .hold(time=40e-6)
        # 6) Move a row up in V: unified “move” with idx selection
        .segment("move_row_up", mode="loop_n", loop=1)
            .tones("V")
            .move(df=-2e6, time=1e-3, idxs=[0])  # idxs defaults to "all" if omitted
        # 8) Ramp off (both logical channels)
        .segment("ramp_off", mode="loop_n", loop=1, phase_mode="optimise")
            .parallel(
                lambda p: p.tones("H")
                .ramp_amp_to(amps=0.0, time=2e-3, kind="adiabatic_ramp")
                .tones("V")
                .ramp_amp_to(amps=0.0, time=2e-3, kind="adiabatic_ramp")
            )
        # 9) Wait (still wrap-continuous; frequency snapping still meaningful)
        .segment("wait_for_trigger_C", mode="wait_trig")
            .hold(time=40e-6)
        # 10) Turn on: state updates at t=0 are allowed; segment length comes from last op
        .segment("turn_on", mode="loop_n", loop=1)
            # .tones("V")
            # .move(df=+2e6, time=0.0)  # state update only
            .tones("H")
            .ramp_amp_to(amps=0.12, time=0.0)  # state update only
            .tones("V")
            .ramp_amp_to(amps=0.4, time=2e-3, kind="adiabatic_ramp")  # gives segment duration
        # 11) Move back (V only)
        .segment("move_up", mode="loop_n", loop=1)
            .tones("V")
            .move(df=(-0.1e6), time=0, idxs=[0])
            .move(df=(-1.9e6), time=2e-3, idxs=[0])
        .segment("wait_for_pullback", mode="wait_trig", phase_mode="optimise")
            .hold(time=40e-6)
        # 12)
        .segment("turn_on_detect", mode="loop_n", loop=1, phase_mode="continue")
            .tones("V")
            .add_tone(f=98.2e6)  # defaults: amp=0.0, phase=0.0
            .ramp_amp_to(idxs=[1], amps=0.4, time=2e-3, kind="adiabatic_ramp")
        # 13) Move back (V only)
        .segment("move_down", mode="loop_n", loop=1, phase_mode="optimise")
            .tones("V")
            .move(df=(+2e6), time=2e-3, idxs=[1])
        # 14) Final wait
        .segment("wait_for_trigger_D", mode="wait_trig", phase_mode="optimise")
            .hold(time=40e-6)
    )

    return ir


_PRESET_BUILDERS: dict[str, Callable[[], AWGProgramBuilder]] = {
    "ramp_down_chirp_2ch": ramp_down_chirp_2ch,
    "simple_tweens": simple_tweens,
    "recreate_mol_exp": recreate_mol_exp,
    "spec_analyser_test": spec_analyser_test,
    "rt_spec_analyser_rearr_hotswap": rt_spec_analyser_rearr_hotswap
}


def available_intent_presets() -> tuple[str, ...]:
    """Return all supported preset keys."""
    return tuple(sorted(_PRESET_BUILDERS))


def build_intent_preset(name_key: str) -> "IntentIR":
    """Build an `IntentIR` from a named preset key."""
    key = str(name_key)
    try:
        builder = _PRESET_BUILDERS[key]()
    except KeyError as exc:
        available = ", ".join(available_intent_presets())
        raise KeyError(
            f"Unknown IntentIR preset {key!r}. Available presets: {available}"
        ) from exc
    return builder.build_intent_ir()
