"""
Example: crest-factor phase optimisation for a multitone waveform.

This is useful for AWG/AOD work: reducing the crest factor of the summed RF
waveform helps avoid amplifier saturation and reduces intermodulation artefacts.

The implementation lives in `awgsegmentfactory.phase_minimiser` so it can be used
from the main library.
"""

from __future__ import annotations

import time

import numpy as np

from awgsegmentfactory.phase_minimiser import (
    MultisineBasis,
    crest_factor,
    minimise_crest_factor_phases,
    schroeder_phases_rad,
    waveform_from_phases,
)


def main() -> None:
    import matplotlib.pyplot as plt

    freqs_MHz = np.array([100, 101, 102, 103, 104, 105, 106], dtype=float)
    amps = np.ones_like(freqs_MHz)

    # Evaluate crest factor over a representative window.
    # If tones are on an integer-MHz grid, 1 µs is a natural period.
    #
    # Choose the time grid so we have at least ~7 samples per period of the
    # highest-frequency tone (avoid being close to Nyquist for crest estimation).
    samples_per_period = 7
    # We'll re-use this same grid later for a sweep up to N=20 tones, so pick the
    # sample rate based on the *largest* frequency in that sweep.
    n_freq_max = 20
    f_max_hz = float((freqs_MHz[0] + (n_freq_max - 1)) * 1e6)
    sample_rate_hz = samples_per_period * f_max_hz
    t_end_s = 3.0e-6
    n_samples = int(np.ceil(t_end_s * sample_rate_hz))
    t_s = np.arange(n_samples, dtype=float) / sample_rate_hz

    basis = MultisineBasis.from_tones(
        t_s=t_s, freqs_hz=(freqs_MHz * 1e6).tolist(), amps=amps.tolist()
    )

    phases_same = np.zeros((basis.n_tones,), dtype=float)
    rng = np.random.default_rng(0)
    phases_random = rng.uniform(0.0, 2.0 * np.pi, size=(basis.n_tones,))

    print("--- analytical (Schroeder) ---")
    start = time.time()
    phases0 = schroeder_phases_rad(basis.n_tones)
    y0 = waveform_from_phases(basis, phases0)
    cf0 = crest_factor(y0)
    print("crest factor", cf0)
    print("time", time.time() - start)

    print("\n--- coordinate refine ---")
    start = time.time()
    phases = minimise_crest_factor_phases(
        basis.freqs_hz,
        basis.amps,
        t_s=basis.t_s,
        passes=1,
        method="coordinate",
    )
    y = waveform_from_phases(basis, phases)
    cf_opt = crest_factor(y)
    print("crest factor", cf_opt)
    print("time", time.time() - start)

    y_same = waveform_from_phases(basis, phases_same)
    y_rand = waveform_from_phases(basis, phases_random)
    cf_same = crest_factor(y_same)
    cf_rand = crest_factor(y_rand)

    print("\n--- baselines ---")
    print("all phases same crest factor", cf_same)
    print("random phases crest factor", cf_rand)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 6))
    axs = axs.ravel()
    for ax, yy, title in [
        (axs[0], y_same, f"All phases same (crest {cf_same:.3f})"),
        (axs[1], y_rand, f"Random phases (crest {cf_rand:.3f})"),
        (axs[2], y0, f"Schroeder seed (crest {cf0:.3f})"),
        (axs[3], y, f"Coordinate refined (crest {cf_opt:.3f})"),
    ]:
        ax.plot(yy, lw=1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Multisine crest-factor phase choices")
    fig.tight_layout()

    # Sweep number of tones: N=1..20 (1 MHz spacing starting at 100 MHz)
    n_freqs = np.arange(1, n_freq_max + 1, dtype=int)
    cf_same_by_n: list[float] = []
    cf_rand_by_n: list[float] = []
    cf_seed_by_n: list[float] = []
    cf_opt_by_n: list[float] = []

    rng_sweep = np.random.default_rng(0)
    random_trials = 20

    for n in n_freqs:
        f_MHz = (freqs_MHz[0] + np.arange(n, dtype=float))
        a = np.ones((n,), dtype=float)
        basis_n = MultisineBasis.from_tones(
            t_s=t_s, freqs_hz=(f_MHz * 1e6).tolist(), amps=a.tolist()
        )

        # Baselines
        y_same_n = waveform_from_phases(basis_n, np.zeros((n,), dtype=float))
        cf_same_by_n.append(crest_factor(y_same_n))

        cf_trials: list[float] = []
        for _ in range(random_trials):
            phases_r = rng_sweep.uniform(0.0, 2.0 * np.pi, size=(n,))
            y_r = waveform_from_phases(basis_n, phases_r)
            cf_trials.append(crest_factor(y_r))
        cf_rand_by_n.append(float(np.mean(cf_trials)))

        # Seed vs refined
        phases_seed_n = schroeder_phases_rad(n)
        y_seed_n = waveform_from_phases(basis_n, phases_seed_n)
        cf_seed_by_n.append(crest_factor(y_seed_n))

        phases_opt_n = minimise_crest_factor_phases(
            basis_n.freqs_hz,
            basis_n.amps,
            t_s=basis_n.t_s,
            passes=1,
            method="coordinate",
        )
        y_opt_n = waveform_from_phases(basis_n, phases_opt_n)
        cf_opt_by_n.append(crest_factor(y_opt_n))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(n_freqs, cf_same_by_n, marker="o", label="All phases same")
    ax2.plot(
        n_freqs,
        cf_rand_by_n,
        marker="o",
        label=f"Random phases (mean of {random_trials})",
    )
    ax2.plot(n_freqs, cf_seed_by_n, marker="o", label="Schroeder seed")
    ax2.plot(n_freqs, cf_opt_by_n, marker="o", label="Coordinate refined")
    ax2.set_xlabel("Number of tones (N)")
    ax2.set_ylabel("Crest factor  max(|y|) / rms(y)")
    ax2.set_title("Crest factor vs number of tones")
    ax2.set_xticks(n_freqs)
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
