import unittest

import numpy as np

from awgsegmentfactory.phase_minimiser import crest_factor, schroeder_phases_rad


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

