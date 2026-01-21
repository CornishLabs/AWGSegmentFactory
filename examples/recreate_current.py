from awgsegmentfactory import AWGProgramBuilder, LinearPositionToFreqCalib
import numpy as np

cal = LinearPositionToFreqCalib(slope_hz_per_um=250e3)  # example: 250 kHz / µm

prog = (
    AWGProgramBuilder(sample_rate=625e6) # 625MHz
    .with_calibration("pos_to_df", cal)
    .plane("H")
    .plane("V")
    # .grid("twz", H="H", V="V")  # optional metadata only; not needed for control

    # Definitions (do not “play” yet)
    .define("loading_H", plane="H",
            freqs=np.linspace(80.0e6, 120.0e6, 12),
            amps=[0.7]*12, phases="auto")
    .define("loading_V", plane="V",
            freqs=[100e6], amps=[0.7], phases="auto")

    .define("exp_H", plane="H",
            freqs=np.linspace(90.0e6, 110.0e6, 8),
            amps=[0.7]*8, phases="auto")
    .define("exp_V", plane="V",
            freqs=[100e6], amps=[0.7], phases="auto")

    # 1) Initial sync: minimum length, wait for first trigger
    .segment("sync", mode="wait_trig")
        .hold(time=1e-9)     # wait_trig defaults: snap_freqs=True, duration rounded to >=1 sample

    # 2) Loading tweezers on: wait for trigger then output steady tones
    .segment("loading_tweezers_on", mode="wait_trig")
        .tones("H").use_def("loading_H")
        .tones("V").use_def("loading_V")
        .hold(time=200e-6, warn_df=50e3)   # warn_df is max |snapping delta| threshold, SI units

    # 3) Rearrange (hotswappable): only H changes (V implicitly unchanged)
    #
    # This is *not* a “move(df=...)” — it’s a remapping/re-targeting operation:
    # “Take selected existing tones, map them onto the target definition ordering,
    #  dropping extras, and tween over time with a chosen curve.”
    .segment("hotswap_rearrange_to_exp_array", mode="loop_n", loop=1)
        .tones("H").remap_from_def(
            target_def="exp_H",
            src=[2, 4, 5, 7, 8, 9, 10, 11],
            dst="all",
            time=2e-4,
            kind="min_jerk",
        )

    # 4) Equalise amps (H only)
    .segment("equalise_amps", mode="loop_n", loop=1)
        .tones("H").ramp_amp_to(
            amps=[0.73, 0.68, 0.67, 0.75, 0.62, 0.81, 0.74, 0.73],
            time=4e-5,
            kind="exp",
            tau=1e-5,
        )

    # 5) Wait for trigger: wrap-continuous quantised hold
    .segment("wait_for_trigger_A", mode="wait_trig")
        .hold(time=200e-6)

    # 6) Move a row up in V: unified “move” with idx selection
    .segment("move_row_up", mode="loop_n", loop=1)
        .tones("V").move(df=2e6, time=1e-6, idxs=[0])  # idxs defaults to "all" if omitted

    # 7) Wait again
    .segment("wait_for_trigger_B", mode="wait_trig")
        .hold(time=200e-6)

    # 8) Ramp off (both planes)
    .segment("ramp_off", mode="loop_n", loop=1)
        .tones("H").ramp_amp_to(amps=0.0, time=4e-5, kind="exp", tau=1e-5)
        .tones("V").ramp_amp_to(amps=0.0, time=4e-5, kind="exp", tau=1e-5)

    # 9) Wait (still wrap-continuous; frequency snapping still meaningful)
    .segment("wait_for_trigger_C", mode="wait_trig")
        .hold(time=200e-6)

    # 10) Turn on: state updates at t=0 are allowed; segment length comes from last op
    .segment("turn_on", mode="loop_n", loop=1)
        .tones("V").move(df=-1e5, time=0.0)               # state update only
        .tones("H").ramp_amp_to(amps=0.7, time=0.0)       # state update only
        .tones("V").ramp_amp_to(amps=0.7, time=4e-5, kind="exp", tau=1e-5)  # gives segment duration

    # 11) Move back (V only)
    .segment("move_back", mode="loop_n", loop=1)
        .tones("V").move(df=(-2e6 + 1e5), time=1e-6, idxs=[0])

    # 12) Final wait
    .segment("wait_for_trigger_D", mode="wait_trig")
        .hold(time=200e-6)

    .build()
)


print(prog.summary())
for seg in prog.segments:
    print("\n---", seg.name, "---")
    print("start tone ids:", [t.tone_id for t in seg.start.tones])
    for p in seg.parts:
        print(p.interp, p.duration_s, "start amps", [t.amp for t in p.start.tones])
