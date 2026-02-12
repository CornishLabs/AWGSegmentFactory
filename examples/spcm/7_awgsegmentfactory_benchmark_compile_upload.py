"""
Benchmark end-to-end latency to "card ready" (armed for trigger):

builder -> resolve -> quantise -> compile (CPU or GPU) -> (optional GPU->CPU copy)
-> CPU upload -> arm card

Usage:
  uv run examples/spcm/7_awgsegmentfactory_benchmark_compile_upload.py
  uv run examples/spcm/7_awgsegmentfactory_benchmark_compile_upload.py --iters 10 --warmup 2
  uv run examples/spcm/7_awgsegmentfactory_benchmark_compile_upload.py --gpu

Notes:
- Upload is currently CPU mode only (host transfer to card memory).
- With `--gpu`, synthesis runs on GPU and this script measures explicit GPU->CPU copy
  before upload, because CPU upload requires NumPy buffers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from awgsegmentfactory import (
    AWGProgramBuilder,
    AWGPhysicalSetupInfo,
    IntentIR,
    QIRtoSamplesSegmentCompiler,
    quantize_resolved_ir,
    resolve_intent_ir,
    upload_sequence_program,
)

from awgsegmentfactory.presets import recreate_mol_exp


@dataclass(frozen=True)
class CompileUploadTiming:
    build_intent_s: float
    intent_codec_s: float
    resolve_s: float
    quantize_s: float
    compile_s: float
    gpu_to_cpu_copy_s: float
    upload_s: float
    arm_ready_s: float
    total_s: float


def _summarize(values_s: Sequence[float]) -> tuple[float, float, float]:
    if not values_s:
        raise ValueError("values_s must be non-empty")
    mn = min(values_s)
    mx = max(values_s)
    mean = sum(values_s) / len(values_s)
    return mn, mean, mx


def _fmt_s(x: float) -> str:
    if x >= 1.0:
        return f"{x:8.3f}s"
    return f"{x * 1e3:8.3f}ms"


def _format_table(runs: Sequence[CompileUploadTiming]) -> str:
    if not runs:
        raise ValueError("runs must be non-empty")

    def col(getter):
        return _summarize([float(getter(r)) for r in runs])

    rows = [
        ("build_intent", col(lambda r: r.build_intent_s)),
        ("intent_codec", col(lambda r: r.intent_codec_s)),
        ("resolve", col(lambda r: r.resolve_s)),
        ("quantize", col(lambda r: r.quantize_s)),
        ("compile", col(lambda r: r.compile_s)),
        ("gpu_to_cpu_copy", col(lambda r: r.gpu_to_cpu_copy_s)),
        ("upload_cpu", col(lambda r: r.upload_s)),
        ("arm_ready", col(lambda r: r.arm_ready_s)),
        ("total", col(lambda r: r.total_s)),
    ]
    out = ["stage            min        mean        max"]
    for name, (mn, mean, mx) in rows:
        out.append(f"{name:<16} {_fmt_s(mn)} {_fmt_s(mean)} {_fmt_s(mx)}")
    return "\n".join(out)


def _build_demo_builder() -> AWGProgramBuilder:
    return recreate_mol_exp()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--sample-rate-hz", type=float, default=625e6)
    p.add_argument("--segment-quantum-s", type=float, default=4e-6)
    p.add_argument("--full-scale-mv", type=float, default=1000.0)
    p.add_argument("--gpu", action="store_true", help="Use CuPy GPU synthesis")
    p.add_argument("--serial-number", type=int, default=None)
    args = p.parse_args()

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.full_scale_mv <= 0:
        raise ValueError("--full-scale-mv must be > 0")

    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})

    if args.gpu:
        import cupy as cp  # type: ignore

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
        name = props.get("name", b"").decode(errors="replace")
        print(f"GPU: {name} (CuPy {cp.__version__}, device {int(dev.id)})")

    import spcm
    from spcm import units

    card_kwargs = {"card_type": spcm.SPCM_TYPE_AO, "verbose": False}
    if args.serial_number is not None:
        card_kwargs["serial_number"] = int(args.serial_number)

    runs: list[CompileUploadTiming] = []

    with spcm.Card(**card_kwargs) as card:
        card.stop(spcm.M2CMD_DATA_STOPDMA)
        card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(float(args.full_scale_mv) * units.mV)
        channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(float(args.sample_rate_hz) * units.Hz)
        clock.clock_output(False)

        full_scale = int(card.max_sample_value()) - 1

        for i in range(args.warmup + args.iters):
            card.stop(spcm.M2CMD_DATA_STOPDMA)
            t0 = perf_counter()

            builder = _build_demo_builder()
            intent = builder.build_intent_ir()
            t1 = perf_counter()

            encoded = intent.encode()
            intent = IntentIR.decode(encoded)
            t2 = perf_counter()

            resolved = resolve_intent_ir(intent, sample_rate_hz=float(args.sample_rate_hz))
            t3 = perf_counter()

            q = quantize_resolved_ir(resolved, segment_quantum_s=float(args.segment_quantum_s))
            t4 = perf_counter()

            output = "cupy" if args.gpu else "numpy"
            slots_compiler = QIRtoSamplesSegmentCompiler(
                quantised=q,
                physical_setup=physical_setup,
                full_scale_mv=float(args.full_scale_mv),
                full_scale=full_scale,
            ).compile_to_card_int16(gpu=bool(args.gpu), output=output)

            if args.gpu:
                import cupy as cp  # type: ignore

                # Compile uses asynchronous kernels; synchronize before stopping timer.
                cp.cuda.Stream.null.synchronize()
            t5 = perf_counter()

            gpu_to_cpu_copy_s = 0.0
            slots_for_upload = slots_compiler
            if args.gpu:
                c0 = perf_counter()
                slots_for_upload = slots_compiler.to_numpy()
                c1 = perf_counter()
                gpu_to_cpu_copy_s = c1 - c0

            u0 = perf_counter()
            upload_sequence_program(slots_for_upload, mode="cpu", card=card, upload_steps=True)
            u1 = perf_counter()

            a0 = perf_counter()
            card.timeout(0)
            # "Card ready" = armed and waiting for trigger.
            card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
            a1 = perf_counter()

            card.stop(spcm.M2CMD_DATA_STOPDMA)
            t6 = perf_counter()

            timing = CompileUploadTiming(
                build_intent_s=t1 - t0,
                intent_codec_s=t2 - t1,
                resolve_s=t3 - t2,
                quantize_s=t4 - t3,
                compile_s=t5 - t4,
                gpu_to_cpu_copy_s=gpu_to_cpu_copy_s,
                upload_s=u1 - u0,
                arm_ready_s=a1 - a0,
                total_s=t6 - t0,
            )
            if i >= args.warmup:
                runs.append(timing)

    print(_format_table(runs))


if __name__ == "__main__":
    main()
