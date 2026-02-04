"""
Fit sin² optical-power calibration models for all calibration files in `examples/calibrations/`.

This utility:
1) routes multiple on-disk calibration formats to the common `OpticalPowerCalCurve` format
2) fits a smooth sin² saturation model with frequency-dependent polynomials
3) writes a small Python module containing `AODSin2Calib` constants that you can import.

Supported file formats (auto-detected by extension):
- `*.txt`  : DE compensation JSON with key `DE_RF_calibration` (freq->amp->DE grid)
- `*.awgde`: JSON dict keyed by optical-power value, each containing (freq, amp) points

Usage:
  .venv/bin/python examples/fit_all_calibrations_in_examples.py
  .venv/bin/python examples/fit_all_calibrations_in_examples.py --output examples/calibrations/sin2_calibration_constants.py
  .venv/bin/python examples/fit_all_calibrations_in_examples.py --no-write  # fit + print metrics only
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from awgsegmentfactory.calibration import AODSin2Calib
from awgsegmentfactory.optical_power_calibration_fit import (
    OpticalPowerCalCurve,
    Sin2PolyFitResult,
    aod_sin2_calib_to_python,
    build_aod_sin2_calib_from_fits,
    curves_from_awgde_dict,
    curves_from_de_rf_calibration_dict,
    fit_sin2_poly_model_by_logical_channel,
    suggest_amp_scale_from_curves,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_curves(path: Path, *, awgde_args: dict) -> tuple[OpticalPowerCalCurve, ...]:
    if path.suffix.lower() == ".awgde":
        cal = _load_json(path)
        if not isinstance(cal, dict):
            raise ValueError(f"{path}: expected top-level JSON object for .awgde")
        curves = curves_from_awgde_dict(cal, **awgde_args)
        if not curves:
            raise ValueError(f"{path}: .awgde produced no usable curves")
        return curves

    cal = _load_json(path)
    de_rf = cal.get("DE_RF_calibration")
    if not isinstance(de_rf, dict):
        raise ValueError(f"{path}: expected JSON key 'DE_RF_calibration' to be a dict")
    curves = curves_from_de_rf_calibration_dict(de_rf)
    if not curves:
        raise ValueError(f"{path}: DE_RF_calibration produced no usable curves")
    return curves


def _group_logical_channels(paths: list[Path]) -> dict[str, dict[str, Path]]:
    """
    Build a dict {dataset_key: {logical_channel: path}}.

    Heuristic:
    - If a filename matches `*_H_calFile_*` or `*_V_calFile_*`, combine H+V into one dataset
      by removing `_(H|V)_calFile` -> `_calFile` from the stem.
    - Otherwise, treat each file as its own dataset with logical_channel="*".
    """
    groups: dict[str, dict[str, Path]] = {}
    hv_re = re.compile(r"^(?P<prefix>.+?)_(?P<lc>[HV])_calFile(?P<rest>_.+)$")
    for p in paths:
        stem = p.stem
        m = hv_re.match(stem)
        if m:
            lc = str(m.group("lc"))
            dataset = f"{m.group('prefix')}_calFile{m.group('rest')}"
            groups.setdefault(dataset, {})[lc] = p
        else:
            groups.setdefault(stem, {})["*"] = p
    return groups


def _py_var_base(dataset_key: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", dataset_key).strip("_").upper()
    if not s:
        s = "CALIB"
    if s[0].isdigit():
        s = "CALIB_" + s
    return s


@dataclass(frozen=True)
class _FitBundle:
    dataset_key: str
    logical_channels: tuple[str, ...]
    source_paths: dict[str, Path]
    curves_by_lc: dict[str, tuple[OpticalPowerCalCurve, ...]]
    fits_by_lc: dict[str, Sin2PolyFitResult]
    calib: AODSin2Calib
    freq_mid_hz: float
    freq_halfspan_hz: float
    amp_scale: float
    rmse_by_lc: dict[str, float]
    max_abs_resid_by_lc: dict[str, float]


def _fit_dataset(
    dataset_key: str,
    paths_by_lc: dict[str, Path],
    *,
    awgde_args: dict,
    degree_g: int,
    degree_v0: int,
    maxiter: int,
) -> _FitBundle:
    curves_by_lc_raw: dict[str, tuple[OpticalPowerCalCurve, ...]] = {}
    for lc, path in paths_by_lc.items():
        curves_by_lc_raw[str(lc)] = _load_curves(path, awgde_args=awgde_args)

    curves_by_lc: dict[str, tuple[OpticalPowerCalCurve, ...]] = {}
    for lc, curves in curves_by_lc_raw.items():
        kept: list[OpticalPowerCalCurve] = []
        for c in curves:
            cc = c.cleaned(clamp_power_nonnegative=True)
            if int(cc.rf_amps_mV.size) == 0:
                continue
            kept.append(cc)
        curves_by_lc[str(lc)] = tuple(kept)

    fits_by_lc, freq_mid_hz, freq_halfspan_hz = fit_sin2_poly_model_by_logical_channel(
        curves_by_lc,
        degree_g=int(degree_g),
        degree_v0=int(degree_v0),
        shared_freq_norm=True,
        maxiter=int(maxiter),
    )
    amp_scale = float(suggest_amp_scale_from_curves(curves_by_lc))
    calib = build_aod_sin2_calib_from_fits(
        fits_by_lc,
        amp_scale=float(amp_scale),
    )
    rmse_by_lc = {lc: float(fit.rmse) for lc, fit in fits_by_lc.items()}
    max_by_lc = {lc: float(fit.max_abs_resid) for lc, fit in fits_by_lc.items()}
    return _FitBundle(
        dataset_key=str(dataset_key),
        logical_channels=tuple(sorted(curves_by_lc.keys())),
        source_paths=dict(paths_by_lc),
        curves_by_lc=curves_by_lc,
        fits_by_lc=fits_by_lc,
        calib=calib,
        freq_mid_hz=float(freq_mid_hz),
        freq_halfspan_hz=float(freq_halfspan_hz),
        amp_scale=float(amp_scale),
        rmse_by_lc=rmse_by_lc,
        max_abs_resid_by_lc=max_by_lc,
    )


def _safe_stem(s: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")
    return out or "fit"


def _plot_fit_bundle(
    bundle: _FitBundle,
    *,
    freq_unit: str,
    n_slices: int,
    save_plots_dir: Path | None,
    save_dpi: int,
    keep_open: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install the `dev` dependency group."
        ) from exc

    from awgsegmentfactory.debug.optical_power_calibration import (
        plot_sin2_fit_parameters,
        plot_sin2_fit_slices,
        plot_sin2_fit_surfaces,
    )

    save_dir = None
    if save_plots_dir is not None:
        save_dir = Path(save_plots_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for lc in bundle.logical_channels:
        curves = bundle.curves_by_lc[lc]
        fit = bundle.fits_by_lc[lc]

        title_base = f"{bundle.dataset_key} [{lc}]"
        fig0, _ = plot_sin2_fit_surfaces(curves, fit, title=title_base, freq_unit=freq_unit)
        fig1, _ = plot_sin2_fit_parameters(fit, title=f"{title_base} parameters", freq_unit=freq_unit)
        fig2, _ = plot_sin2_fit_slices(
            curves,
            fit,
            n_slices=int(n_slices),
            title=f"{title_base} slices",
            freq_unit=freq_unit,
        )

        if save_dir is not None:
            stem = _safe_stem(f"{bundle.dataset_key}__{lc}")
            fig0.savefig(save_dir / f"{stem}__surfaces.png", dpi=int(save_dpi))
            fig1.savefig(save_dir / f"{stem}__params.png", dpi=int(save_dpi))
            fig2.savefig(save_dir / f"{stem}__slices.png", dpi=int(save_dpi))

        if not keep_open:
            import matplotlib.pyplot as plt

            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)


def _write_constants_file(output: Path, bundles: list[_FitBundle]) -> None:
    lines: list[str] = []
    lines.append('"""Generated optical-power calibration constants.')
    lines.append("")
    lines.append("Generated by `examples/fit_all_calibrations_in_examples.py`.")
    lines.append("Model: sin2")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from awgsegmentfactory.calibration import AODSin2Calib")
    lines.append("")

    names: dict[str, str] = {}
    for b in bundles:
        var = f"{_py_var_base(b.dataset_key)}_SIN2_CALIB"
        names[b.dataset_key] = var
        src = ", ".join(f"{lc}={p.as_posix()!r}" for lc, p in sorted(b.source_paths.items()))
        lines.append(f"# source: {src}")
        lines.append(aod_sin2_calib_to_python(b.calib, var_name=var))
        lines.append("")

    lines.append("CALIBS_BY_DATASET_KEY: dict[str, AODSin2Calib] = {")
    for key in sorted(names.keys()):
        lines.append(f"    {key!r}: {names[key]},")
    lines.append("}")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib-dir",
        default="examples/calibrations",
        help="Directory containing calibration files to fit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the generated Python constants module (defaults to examples/calibrations/sin2_calibration_constants.py).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write the output module (fit + print metrics only).",
    )
    parser.add_argument("--degree-g", type=int, default=6)
    parser.add_argument("--degree-v0", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=250)

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Show matplotlib plots for each fitted dataset (requires matplotlib).",
    )
    parser.add_argument(
        "--save-plots-dir",
        default=None,
        help="If set, save plots as PNGs to this directory (still requires matplotlib).",
    )
    parser.add_argument(
        "--plot-freq-unit",
        choices=("Hz", "kHz", "MHz"),
        default="MHz",
        help="Frequency unit used on plot axes.",
    )
    parser.add_argument(
        "--plot-n-slices",
        type=int,
        default=7,
        help="Number of per-frequency slices in the amp-vs-power plot.",
    )
    parser.add_argument(
        "--plot-save-dpi",
        type=int,
        default=160,
        help="DPI used when saving plots.",
    )

    # `.awgde` adapter controls (for speed).
    parser.add_argument("--awgde-freq-round-mhz", type=float, default=1.0)
    parser.add_argument("--awgde-max-power-levels", type=int, default=300)
    parser.add_argument("--awgde-min-points-per-curve", type=int, default=20)
    parser.add_argument("--awgde-max-points-per-curve", type=int, default=250)
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    if not calib_dir.exists():
        raise SystemExit(f"Calibration dir not found: {calib_dir}")

    files: list[Path] = []
    files.extend(sorted(calib_dir.glob("*.txt")))
    files.extend(sorted(calib_dir.glob("*.awgde")))
    if not files:
        raise SystemExit(f"No calibration files found in {calib_dir}")

    groups = _group_logical_channels(files)
    awgde_args = {
        "freq_round_hz": float(args.awgde_freq_round_mhz) * 1e6,
        "max_power_levels": int(args.awgde_max_power_levels) if args.awgde_max_power_levels else None,
        "min_points_per_curve": int(args.awgde_min_points_per_curve),
        "max_points_per_curve": int(args.awgde_max_points_per_curve)
        if args.awgde_max_points_per_curve
        else None,
    }

    want_plots = bool(args.plots) or (args.save_plots_dir is not None)
    if want_plots:
        # Validate early so we error once, before long fits.
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "This script's plotting mode requires matplotlib. Install the `dev` dependency group."
            ) from exc

    bundles: list[_FitBundle] = []
    for dataset_key, paths_by_lc in sorted(groups.items()):
        print(f"\n=== {dataset_key} ===")
        for lc, p in sorted(paths_by_lc.items()):
            print(f"  input[{lc}]: {p}")

        b = _fit_dataset(
            dataset_key,
            paths_by_lc,
            awgde_args=awgde_args,
            degree_g=int(args.degree_g),
            degree_v0=int(args.degree_v0),
            maxiter=int(args.maxiter),
        )
        print(f"  logical_channels: {b.logical_channels}")
        print(f"  freq_mid_hz: {b.freq_mid_hz:.6g}")
        print(f"  freq_halfspan_hz: {b.freq_halfspan_hz:.6g}")
        print(f"  amp_scale: {b.amp_scale:.6g}")
        for lc in b.logical_channels:
            print(f"  [{lc}] rmse={b.rmse_by_lc[lc]:.4g}  max|res|={b.max_abs_resid_by_lc[lc]:.4g}")
        if want_plots:
            _plot_fit_bundle(
                b,
                freq_unit=str(args.plot_freq_unit),
                n_slices=int(args.plot_n_slices),
                save_plots_dir=Path(args.save_plots_dir) if args.save_plots_dir else None,
                save_dpi=int(args.plot_save_dpi),
                keep_open=bool(args.plots),
            )
        bundles.append(b)

    if args.save_plots_dir:
        print(f"\nSaved plots to: {args.save_plots_dir}")

    if args.no_write:
        if args.plots:
            import matplotlib.pyplot as plt

            plt.show()
        return

    out = Path(args.output) if args.output else Path("examples/calibrations/sin2_calibration_constants.py")
    _write_constants_file(out, bundles)
    print(f"\nWrote: {out}")
    if args.plots:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
