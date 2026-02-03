import unittest

import numpy as np

from awgsegmentfactory.calibration import AODTanh2Calib
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    aod_tanh2_calib_to_python,
    build_aod_tanh2_calib_from_fits,
    fit_tanh2_poly_model,
    fit_tanh2_poly_model_by_logical_channel,
    suggest_amp_scale_from_curves,
)


class TestOpticalPowerCalibrationFit(unittest.TestCase):
    def test_fit_tanh2_recovers_constant_params(self) -> None:
        freqs_hz = np.array([90e6, 100e6, 110e6], dtype=float)
        amps_mV = np.linspace(0.0, 200.0, 48, dtype=float)
        g_true = 0.8
        v0_true = 75.0
        min_v0_sq = 1e-12

        curves = []
        for f in freqs_hz:
            p = float(g_true) * (np.tanh(amps_mV / float(v0_true)) ** 2)
            curves.append(OpticalPowerCalCurve(freq_hz=float(f), rf_amps_mV=amps_mV, optical_powers=p))

        fit = fit_tanh2_poly_model(
            curves,
            degree_g=0,
            degree_v0=0,
            min_v0_sq_mV2=min_v0_sq,
            maxiter=80,
        )

        self.assertLess(fit.rmse, 1e-9)
        g_est = float(fit.g_of_freq(np.array([100e6], dtype=float))[0])
        v0_est = float(fit.v0_of_freq_mV(np.array([100e6], dtype=float))[0])
        self.assertAlmostEqual(g_est, g_true, places=9)
        self.assertAlmostEqual(v0_est, v0_true, places=6)

    def test_multi_channel_bundle_and_python_snippet(self) -> None:
        freqs_hz = np.array([90e6, 100e6, 110e6], dtype=float)
        amps_mV = np.linspace(0.0, 180.0, 40, dtype=float)
        min_v0_sq = 1e-12

        curves_by_lc = {}
        for lc, (g_true, v0_true) in {"H": (0.9, 60.0), "V": (0.7, 90.0)}.items():
            curves = []
            for f in freqs_hz:
                p = float(g_true) * (np.tanh(amps_mV / float(v0_true)) ** 2)
                curves.append(
                    OpticalPowerCalCurve(freq_hz=float(f), rf_amps_mV=amps_mV, optical_powers=p)
                )
            curves_by_lc[lc] = tuple(curves)

        fits_by_lc, _mid, _halfspan = fit_tanh2_poly_model_by_logical_channel(
            curves_by_lc,
            degree_g=0,
            degree_v0=0,
            shared_freq_norm=True,
            min_v0_sq_mV2=min_v0_sq,
            maxiter=80,
        )
        amp_scale = suggest_amp_scale_from_curves(curves_by_lc)
        calib = build_aod_tanh2_calib_from_fits(fits_by_lc, amp_scale=float(amp_scale))
        self.assertIsInstance(calib, AODTanh2Calib)
        self.assertEqual(set(calib.g_poly_by_logical_channel.keys()), {"H", "V"})
        self.assertEqual(set(calib.v0_a_poly_by_logical_channel.keys()), {"H", "V"})

        code = aod_tanh2_calib_to_python(calib, var_name="CAL")
        self.assertIn("CAL = AODTanh2Calib(", code)
        self.assertIn("'H':", code)
        self.assertIn("'V':", code)

