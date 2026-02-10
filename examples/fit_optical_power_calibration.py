"""
Example wrapper for the optical-power calibration tool.

Equivalent CLI invocation:
  python -m awgsegmentfactory.tools.fit_optical_power_calibration --input-data-file examples/calibrations/814_H_calFile_17.02.2022_0=0.txt --input-data-format de-json --plot
"""

from __future__ import annotations

from pathlib import Path

from awgsegmentfactory.tools.fit_optical_power_calibration import main as tool_main


def _default_calibration_file() -> Path:
    p = Path("examples/calibrations/814_H_calFile_17.02.2022_0=0.txt")
    if p.exists():
        return p
    return Path("examples/814_H_calFile_17.02.2022_0=0.txt")


def main() -> None:
    tool_main(
        [
            "--input-data-file",
            _default_calibration_file().as_posix(),
            "--input-data-format",
            "de-json",
            "--plot",
        ]
    )


if __name__ == "__main__":
    main()
