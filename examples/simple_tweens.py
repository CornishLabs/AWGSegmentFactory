from awgsegmentfactory import AWGProgramBuilder

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
    .build_timeline(sample_rate_hz=fs)
)
