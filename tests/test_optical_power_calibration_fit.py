import unittest

import numpy as np

from awgsegmentfactory.calibration import AODSin2Calib
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    aod_sin2_calib_to_python,
    fit_sin2_poly_model,
    fit_sin2_poly_model_by_logical_channel,
    suggest_amp_scale_from_curves,
)


class TestOpticalPowerCalibrationFit(unittest.TestCase):
    def test_fit_sin2_recovers_constant_params(self) -> None:
        freqs_hz = np.array([90e6, 100e6, 110e6], dtype=float)
        v0_true = 150.0
        amps_mV = np.linspace(0.0, v0_true, 48, dtype=float)
        g_true = 0.8
        min_v0_sq = 1e-12

        curves = []
        for f in freqs_hz:
            p = float(g_true) * (np.sin((0.5 * np.pi) * (amps_mV / float(v0_true))) ** 2)
            curves.append(OpticalPowerCalCurve(freq_hz=float(f), rf_amps_mV=amps_mV, optical_powers=p))

        fit = fit_sin2_poly_model(
            curves,
            degree_g=0,
            degree_v0=0,
            min_v0_sq_mV2=min_v0_sq,
            maxiter=120,
        )

        self.assertLess(fit.rmse, 1e-9)
        g_est = float(fit.g_of_freq(np.array([100e6], dtype=float))[0])
        v0_est = float(fit.v0_of_freq_mV(np.array([100e6], dtype=float))[0])
        self.assertAlmostEqual(g_est, g_true, places=9)
        self.assertAlmostEqual(v0_est, v0_true, places=6)

    def test_per_channel_fit_and_python_snippet(self) -> None:
        freqs_hz = np.array([90e6, 100e6, 110e6], dtype=float)
        amps_mV = np.linspace(0.0, 150.0, 40, dtype=float)
        min_v0_sq = 1e-12

        curves_by_lc = {}
        for lc, (g_true, v0_true) in {"H": (0.9, 140.0), "V": (0.7, 180.0)}.items():
            curves = []
            for f in freqs_hz:
                p = float(g_true) * (np.sin((0.5 * np.pi) * (amps_mV / float(v0_true))) ** 2)
                curves.append(
                    OpticalPowerCalCurve(freq_hz=float(f), rf_amps_mV=amps_mV, optical_powers=p)
                )
            curves_by_lc[lc] = tuple(curves)

        fits_by_lc, _mid, _halfspan = fit_sin2_poly_model_by_logical_channel(
            curves_by_lc,
            degree_g=0,
            degree_v0=0,
            shared_freq_norm=True,
            min_v0_sq_mV2=min_v0_sq,
            maxiter=120,
        )
        amp_scale = suggest_amp_scale_from_curves(curves_by_lc)
        cal_h = fits_by_lc["H"].to_aod_sin2_calib(
            amp_scale=float(amp_scale),
            freq_min_hz=float(np.min(freqs_hz)),
            freq_max_hz=float(np.max(freqs_hz)),
            traceability_string="H curve set",
        )
        cal_v = fits_by_lc["V"].to_aod_sin2_calib(
            amp_scale=float(amp_scale),
            freq_min_hz=float(np.min(freqs_hz)),
            freq_max_hz=float(np.max(freqs_hz)),
            traceability_string="V curve set",
        )
        self.assertIsInstance(cal_h, AODSin2Calib)
        self.assertIsInstance(cal_v, AODSin2Calib)
        self.assertGreaterEqual(int(cal_h.best_freq_hz), int(np.min(freqs_hz)))
        self.assertLessEqual(int(cal_h.best_freq_hz), int(np.max(freqs_hz)))

        code = aod_sin2_calib_to_python(cal_h, var_name="CAL_H")
        self.assertIn("CAL_H = AODSin2Calib(", code)
        self.assertIn("g_poly_high_to_low=", code)
        self.assertIn("v0_a_poly_high_to_low=", code)
