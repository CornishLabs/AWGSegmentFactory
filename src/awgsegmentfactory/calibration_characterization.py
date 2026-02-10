"""Thin aliases for serialized AWG calibration objects."""

from __future__ import annotations

from .calibration import AODSin2Calib, AWGCalibration

# Historical names retained as simple aliases.
CalibrationCharacterization = AWGCalibration
Calibration = AWGCalibration

__all__ = [
    "AODSin2Calib",
    "AWGCalibration",
    "CalibrationCharacterization",
    "Calibration",
]
