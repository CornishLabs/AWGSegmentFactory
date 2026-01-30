import unittest

import numpy as np

from awgsegmentfactory.phase_minimiser import crest_factor, minimise_crest_factor_phases, schroeder_phases_rad


class TestPhaseMinimiser(unittest.TestCase):
    def test_crest_factor_uses_abs_peak(self) -> None:
        # If crest used max(y) instead of max(abs(y)), this would go negative.
        y = np.array([-2.0, -1.0], dtype=float)
        got = crest_factor(y)
        expected = float(np.max(np.abs(y)) / np.sqrt(np.mean(y * y)))
        self.assertAlmostEqual(got, expected)
        self.assertGreater(got, 0.0)

    def test_schroeder_seed_shape_and_range(self) -> None:
        phases = schroeder_phases_rad(7)
        self.assertEqual(phases.shape, (7,))
        self.assertAlmostEqual(float(phases[0]), 0.0)
        self.assertTrue(np.all(phases >= 0.0))
        self.assertTrue(np.all(phases < 2.0 * np.pi + 1e-12))

    def test_fixed_mask_keeps_fixed_phases(self) -> None:
        freqs_hz = [10.0, 20.0]
        amps = [1.0, 1.0]
        t_s = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=float)
        phases_init = np.array([0.123, 0.456], dtype=float)
        fixed_mask = np.array([True, False])

        phases_out = minimise_crest_factor_phases(
            freqs_hz,
            amps,
            t_s=t_s,
            phases_init_rad=phases_init,
            fixed_mask=fixed_mask,
            passes=1,
            method="coordinate",
            output="rad",
        )
        self.assertAlmostEqual(float(phases_out[0]), float(phases_init[0]), places=12)

    def test_all_zero_amps_returns_init(self) -> None:
        freqs_hz = [10.0, 20.0]
        amps = [0.0, 0.0]
        phases_init = np.array([0.1, 0.2], dtype=float)
        t_s = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=float)

        phases_out = minimise_crest_factor_phases(
            freqs_hz,
            amps,
            t_s=t_s,
            phases_init_rad=phases_init,
            passes=1,
            method="coordinate",
            output="rad",
        )
        np.testing.assert_allclose(phases_out, phases_init)
