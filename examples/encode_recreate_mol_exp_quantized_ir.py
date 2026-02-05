"""
Build + quantize `recreate_mol_exp()` and print the encoded QuantizedIR dictionary.

Run:
  uv run examples/encode_recreate_mol_exp_quantized_ir.py

Optional:
  uv run examples/encode_recreate_mol_exp_quantized_ir.py --out encoded.pkl
"""

from __future__ import annotations

import argparse
import pickle
import pprint
from pathlib import Path

from awgsegmentfactory.quantize import quantize_resolved_ir

from sequence_repo import recreate_mol_exp


def _parse_channel_map(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        if "=" not in x:
            raise ValueError(f"Invalid --map entry {x!r}; expected like H=0")
        k, v = x.split("=", 1)
        out[str(k)] = int(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=float, default=625e6, help="Sample rate in Hz.")
    parser.add_argument(
        "--map",
        nargs="*",
        default=["H=0", "V=1"],
        help="Logical->hardware channel map entries like H=0 V=1.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output path to write a pickle of the encoded dict.",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Print with plain `print(encoded)` instead of pprint.",
    )
    args = parser.parse_args()

    builder = recreate_mol_exp()
    ir = builder.build_resolved_ir(sample_rate_hz=float(args.fs))
    q = quantize_resolved_ir(ir, logical_channel_to_hardware_channel=_parse_channel_map(args.map))
    encoded = q.encode()

    if args.out:
        out_path = Path(args.out)
        with out_path.open("wb") as f:
            pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote pickle: {out_path}")

    if args.no_pretty:
        print(encoded)
    else:
        pprint.pprint(encoded, width=120, compact=True, sort_dicts=False)


if __name__ == "__main__":
    main()

