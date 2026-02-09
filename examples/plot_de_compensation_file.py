"""
Compatibility wrapper for the merged calibration example.

`plot_de_compensation_file.py` and `fit_optical_power_calibration.py` were merged.
Use:

  python examples/fit_optical_power_calibration.py --input H=<path>

This wrapper preserves the old entrypoint and defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fit_optical_power_calibration import main as _merged_main


def _default_path() -> str:
    p = Path("examples/calibrations/814_H_calFile_17.02.2022_0=0.txt")
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("path", nargs="?")
    args, rest = parser.parse_known_args()

    path = args.path if args.path is not None else _default_path()
    print(
        "Note: this example has moved into `examples/fit_optical_power_calibration.py`."
    )

    sys.argv = [
        sys.argv[0],
        "--input",
        f"H={path}",
        "--overview-plots",
        *rest,
    ]
    _merged_main()


if __name__ == "__main__":
    main()
