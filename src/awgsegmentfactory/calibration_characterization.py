"""Serializable characterization payload for fitted optical-power calibrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .calibration import AODSin2Calib, MultiChannelAODSin2Calib
from .optical_power_calibration_fit import OpticalPowerCalCurve, Sin2PolyFitResult

_SCHEMA = "awgsegmentfactory.sin2_characterization.v2"


def _aod_sin2_calib_to_dict(calib: AODSin2Calib) -> Dict[str, Any]:
    return {
        "type": "AODSin2Calib",
        "g_poly_high_to_low": [float(x) for x in tuple(calib.g_poly_high_to_low)],
        "v0_a_poly_high_to_low": [float(x) for x in tuple(calib.v0_a_poly_high_to_low)],
        "freq_mid_hz": float(calib.freq_mid_hz),
        "freq_halfspan_hz": float(calib.freq_halfspan_hz),
        "amp_scale": float(calib.amp_scale),
        "min_g": float(calib.min_g),
        "min_v0_sq": float(calib.min_v0_sq),
        "y_eps": float(calib.y_eps),
    }


def _aod_sin2_calib_from_dict(data: Mapping[str, Any]) -> AODSin2Calib:
    return AODSin2Calib(
        g_poly_high_to_low=tuple(float(x) for x in list(data["g_poly_high_to_low"])),
        v0_a_poly_high_to_low=tuple(float(x) for x in list(data["v0_a_poly_high_to_low"])),
        freq_mid_hz=float(data["freq_mid_hz"]),
        freq_halfspan_hz=float(data["freq_halfspan_hz"]),
        amp_scale=float(data["amp_scale"]),
        min_g=float(data["min_g"]),
        min_v0_sq=float(data["min_v0_sq"]),
        y_eps=float(data["y_eps"]),
    )


def _logical_index_map(data: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in data.items():
        out[str(k)] = int(v)
    return out


@dataclass(frozen=True)
class ChannelCalibrationCharacterization:
    """
    Per-physical-channel characterization entry.

    This object is stored in a list where list index == physical channel index.
    """

    freq_min_hz: float
    freq_max_hz: float
    highest_de_freq_hz: float
    highest_de_value: float
    rmse: float
    max_abs_resid: float
    source_file: str
    source_format: str
    calib: AODSin2Calib

    def serialise(self) -> Dict[str, Any]:
        return {
            "freq_min_hz": float(self.freq_min_hz),
            "freq_max_hz": float(self.freq_max_hz),
            "highest_de_freq_hz": float(self.highest_de_freq_hz),
            "highest_de_value": float(self.highest_de_value),
            "rmse": float(self.rmse),
            "max_abs_resid": float(self.max_abs_resid),
            "source_file": str(self.source_file),
            "source_format": str(self.source_format),
            "calibration": _aod_sin2_calib_to_dict(self.calib),
        }

    def serialize(self) -> Dict[str, Any]:
        return self.serialise()

    @classmethod
    def deserialise(cls, data: Mapping[str, Any]) -> "ChannelCalibrationCharacterization":
        calib_raw = data.get("calibration")
        if not isinstance(calib_raw, Mapping):
            raise ValueError("Channel characterization missing 'calibration'")
        return cls(
            freq_min_hz=float(data["freq_min_hz"]),
            freq_max_hz=float(data["freq_max_hz"]),
            highest_de_freq_hz=float(data["highest_de_freq_hz"]),
            highest_de_value=float(data["highest_de_value"]),
            rmse=float(data["rmse"]),
            max_abs_resid=float(data["max_abs_resid"]),
            source_file=str(data.get("source_file", "")),
            source_format=str(data.get("source_format", "unknown")),
            calib=_aod_sin2_calib_from_dict(calib_raw),
        )

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> "ChannelCalibrationCharacterization":
        return cls.deserialise(data)


@dataclass(frozen=True)
class CalibrationCharacterization:
    """
    Persisted multi-channel calibration characterization.

    - `channels[i]` corresponds to physical channel index `i`.
    - `logical_to_channel_index` maps logical names to those indices.
    """

    logical_to_channel_index: Dict[str, int]
    channels: list[ChannelCalibrationCharacterization]
    freq_min_hz: float
    freq_max_hz: float
    highest_de_channel_index: int
    highest_de_freq_hz: float
    highest_de_value: float
    schema: str = _SCHEMA
    model: str = "sin2_poly"

    def __post_init__(self) -> None:
        logical_map = {str(k): int(v) for k, v in dict(self.logical_to_channel_index).items()}
        channels = list(self.channels)
        if not logical_map:
            raise ValueError("logical_to_channel_index must not be empty")
        if not channels:
            raise ValueError("channels list must not be empty")
        n = len(channels)
        for logical, idx in logical_map.items():
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"logical_to_channel_index[{logical!r}]={idx} out of range for {n} channels"
                )
        if set(logical_map.values()) != set(range(n)):
            raise ValueError(
                "logical_to_channel_index must cover each channel index exactly once "
                f"(expected indices 0..{n - 1})"
            )
        if int(self.highest_de_channel_index) < 0 or int(self.highest_de_channel_index) >= n:
            raise ValueError("highest_de_channel_index out of range")

        object.__setattr__(self, "logical_to_channel_index", logical_map)
        object.__setattr__(self, "channels", channels)

    @classmethod
    def from_fit(
        cls,
        *,
        channel_calibs_by_logical_channel: Mapping[str, AODSin2Calib],
        curves_by_logical_channel: Mapping[str, Sequence[OpticalPowerCalCurve]],
        fits_by_logical_channel: Mapping[str, Sin2PolyFitResult],
        logical_to_channel_index: Mapping[str, int],
        source_files_by_logical_channel: Mapping[str, str | Path] | None = None,
        source_format_by_logical_channel: Mapping[str, str] | None = None,
        peak_scan_points: int = 4096,
    ) -> "CalibrationCharacterization":
        mapping = {str(k): int(v) for k, v in dict(logical_to_channel_index).items()}
        if not mapping:
            raise ValueError("logical_to_channel_index must not be empty")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("logical_to_channel_index must map each logical channel to a unique index")
        indices = sorted(set(mapping.values()))
        if indices != list(range(len(indices))):
            raise ValueError(
                "logical_to_channel_index indices must be contiguous from 0 "
                f"(got {indices})"
            )

        src_files_raw = source_files_by_logical_channel or {}
        src_format_raw = source_format_by_logical_channel or {}

        n_channels = len(indices)
        channels: list[ChannelCalibrationCharacterization | None] = [None] * n_channels
        n_eval = max(32, int(peak_scan_points))

        for logical, idx in mapping.items():
            curves = curves_by_logical_channel.get(logical)
            fit = fits_by_logical_channel.get(logical)
            calib = channel_calibs_by_logical_channel.get(logical)
            if curves is None:
                raise KeyError(f"Missing curves for logical_channel {logical!r}")
            if fit is None:
                raise KeyError(f"Missing fit for logical_channel {logical!r}")
            if calib is None:
                raise KeyError(f"Missing calibration for logical_channel {logical!r}")

            ff = np.asarray([float(c.freq_hz) for c in curves], dtype=float)
            if ff.size == 0:
                raise ValueError(f"No frequency points for logical_channel {logical!r}")
            freq_min = float(np.min(ff))
            freq_max = float(np.max(ff))

            f_grid = np.linspace(freq_min, freq_max, num=n_eval, dtype=float)
            g_eval = np.asarray(fit.g_of_freq(f_grid), dtype=float)
            finite = np.isfinite(g_eval)
            if not np.any(finite):
                raise ValueError(
                    f"fit.g_of_freq produced no finite values for logical_channel {logical!r}"
                )
            g_safe = np.where(finite, g_eval, -np.inf)
            i_peak = int(np.argmax(g_safe))
            peak_freq = float(f_grid[i_peak])
            peak_val = float(g_safe[i_peak])

            raw_path = src_files_raw.get(logical)
            source_file = str(Path(raw_path).as_posix()) if raw_path is not None else ""
            source_format = str(src_format_raw.get(logical, "unknown"))

            channels[int(idx)] = ChannelCalibrationCharacterization(
                freq_min_hz=freq_min,
                freq_max_hz=freq_max,
                highest_de_freq_hz=peak_freq,
                highest_de_value=peak_val,
                rmse=float(fit.rmse),
                max_abs_resid=float(fit.max_abs_resid),
                source_file=source_file,
                source_format=source_format,
                calib=calib,
            )

        if any(ch is None for ch in channels):
            raise ValueError("Some physical channel indices were not populated")
        channels_typed = [ch for ch in channels if ch is not None]

        freq_min_hz = float(min(ch.freq_min_hz for ch in channels_typed))
        freq_max_hz = float(max(ch.freq_max_hz for ch in channels_typed))
        highest_de_channel_index = int(
            np.argmax(np.asarray([ch.highest_de_value for ch in channels_typed], dtype=float))
        )
        highest_de_freq_hz = float(channels_typed[highest_de_channel_index].highest_de_freq_hz)
        highest_de_value = float(channels_typed[highest_de_channel_index].highest_de_value)

        return cls(
            logical_to_channel_index=mapping,
            channels=channels_typed,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            highest_de_channel_index=highest_de_channel_index,
            highest_de_freq_hz=highest_de_freq_hz,
            highest_de_value=highest_de_value,
        )

    def serialise(self) -> Dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model": str(self.model),
            "logical_to_channel_index": {
                str(k): int(v) for k, v in self.logical_to_channel_index.items()
            },
            "channels": [ch.serialise() for ch in self.channels],
            "freq_min_hz": float(self.freq_min_hz),
            "freq_max_hz": float(self.freq_max_hz),
            "highest_de_channel_index": int(self.highest_de_channel_index),
            "highest_de_freq_hz": float(self.highest_de_freq_hz),
            "highest_de_value": float(self.highest_de_value),
        }

    def serialize(self) -> Dict[str, Any]:
        return self.serialise()

    @classmethod
    def deserialise(cls, data: Mapping[str, Any]) -> "CalibrationCharacterization":
        schema = str(data.get("schema", _SCHEMA))
        if schema != _SCHEMA:
            raise ValueError(f"Unsupported schema {schema!r}; expected {_SCHEMA!r}")

        channels_raw = data.get("channels")
        if not isinstance(channels_raw, list) or not channels_raw:
            raise ValueError("Characterization must contain non-empty 'channels' list")
        channels = [ChannelCalibrationCharacterization.deserialise(ch) for ch in channels_raw]

        logical_map_raw = data.get("logical_to_channel_index")
        if not isinstance(logical_map_raw, Mapping):
            raise ValueError("Characterization must contain 'logical_to_channel_index' mapping")

        return cls(
            logical_to_channel_index=_logical_index_map(logical_map_raw),
            channels=channels,
            freq_min_hz=float(data["freq_min_hz"]),
            freq_max_hz=float(data["freq_max_hz"]),
            highest_de_channel_index=int(data["highest_de_channel_index"]),
            highest_de_freq_hz=float(data["highest_de_freq_hz"]),
            highest_de_value=float(data["highest_de_value"]),
            schema=schema,
            model=str(data.get("model", "sin2_poly")),
        )

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> "CalibrationCharacterization":
        return cls.deserialise(data)

    def to_file(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.serialise(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "CalibrationCharacterization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Characterization file must contain a JSON object")
        return cls.deserialise(payload)

    def to_multi_channel_calib(self) -> MultiChannelAODSin2Calib:
        return MultiChannelAODSin2Calib(
            channel_calibs=tuple(ch.calib for ch in self.channels),
            logical_to_channel_index=dict(self.logical_to_channel_index),
        )


Calibration = CalibrationCharacterization

__all__ = [
    "ChannelCalibrationCharacterization",
    "CalibrationCharacterization",
    "Calibration",
]

