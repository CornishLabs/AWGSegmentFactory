"""
Benchmark / profile compilation time from a program definition to final int16 samples.

Usage:
  python examples/benchmark_pipeline.py
  python examples/benchmark_pipeline.py --iters 10 --warmup 2
  python examples/benchmark_pipeline.py --gpu
  python examples/benchmark_pipeline.py --profile --profile-out profile.prof

Notes:
  --gpu only affects the final "compile" stage (sample synthesis). Resolve/quantize
  still run on CPU. The final int16 buffers are transferred back to CPU/NumPy.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats

from awgsegmentfactory import AWGProgramBuilder
from awgsegmentfactory.calibration import AWGPhysicalSetupInfo
from awgsegmentfactory.debug import (
    benchmark_builder_pipeline,
    compile_builder_pipeline_timed,
    format_benchmark_table,
)


from awgsegmentfactory.presets import recreate_mol_exp


def build_demo_builder() -> AWGProgramBuilder:
    return recreate_mol_exp()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--sample-rate-hz", type=float, default=625e6)
    p.add_argument("--gpu", action="store_true", help="Use CuPy GPU synthesis")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-out", type=str, default=None)
    p.add_argument("--profile-sort", type=str, default="cumtime")
    p.add_argument("--profile-top", type=int, default=30)
    args = p.parse_args()

    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})

    if args.gpu:
        import cupy as cp  # type: ignore

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
        name = props.get("name", b"").decode(errors="replace")
        print(f"GPU: {name} (CuPy {cp.__version__}, device {int(dev.id)})")

    runs = benchmark_builder_pipeline(
        build_demo_builder,
        sample_rate_hz=args.sample_rate_hz,
        physical_setup=physical_setup,
        iters=args.iters,
        warmup=args.warmup,
        gpu=args.gpu,
    )
    print(format_benchmark_table(runs))

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        _compiled, timing = compile_builder_pipeline_timed(
            build_demo_builder(),
            sample_rate_hz=args.sample_rate_hz,
            physical_setup=physical_setup,
            gpu=args.gpu,
        )
        prof.disable()
        print("\nSingle-run timings:")
        print(timing)

        if args.profile_out:
            prof.dump_stats(args.profile_out)
            print(f"\nWrote profile to: {args.profile_out}")

        s = io.StringIO()
        stats = pstats.Stats(prof, stream=s).strip_dirs().sort_stats(args.profile_sort)
        stats.print_stats(args.profile_top)
        print("\nTop functions:")
        print(s.getvalue())


if __name__ == "__main__":
    main()
