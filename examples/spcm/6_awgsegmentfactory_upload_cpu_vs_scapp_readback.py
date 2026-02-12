"""
Verify sequence-memory equivalence between CPU upload and SCAPP (GPU) upload.

Workflow:
1) Build one QuantizedIR.
2) Compile + upload with CPU (`mode="cpu"`), then read back with CPU.
3) Compile + upload with SCAPP (`mode="scapp"`), then read back with CPU.
4) Compare per-segment int16 buffers.

Requirements:
- Spectrum card with sequence mode and SCAPP option for the GPU path.
- CuPy + compatible CUDA runtime for SCAPP upload.
- The card should be idle while readback is performed.
"""

from __future__ import annotations

import numpy as np

from awgsegmentfactory import (
    AWGPhysicalSetupInfo,
    AWGProgramBuilder,
    QIRtoSamplesSegmentCompiler,
    quantize_resolved_ir,
    readback_sequence_segments_to_numpy,
    upload_sequence_program,
)


def _build_quantised(sample_rate_hz: float):
    b = (
        AWGProgramBuilder()
        .logical_channel("H")
        .logical_channel("V")
        .define("init_H", logical_channel="H", freqs=[1.1e5], amps=[300.0], phases="auto")
        .define("init_V", logical_channel="V", freqs=[2.2e5], amps=[240.0], phases="auto")
    )
    b.segment("wait", mode="wait_trig")
    b.tones("H").use_def("init_H")
    b.tones("V").use_def("init_V")
    b.hold(time=80e-6)

    b.segment("move", mode="once")
    b.tones("H").move(df=+0.8e5, time=120e-6, kind="linear")
    b.tones("V").ramp_amp_to(amps=[80.0], time=120e-6, kind="exp", tau=50e-6)

    ir = b.build_resolved_ir(sample_rate_hz=sample_rate_hz)
    return quantize_resolved_ir(ir, segment_quantum_s=4e-6)


def _to_segment_dict(items):
    return {int(i): np.asarray(data, dtype=np.int16) for i, data in items}


def main() -> None:
    import spcm
    from spcm import units

    try:
        import cupy as cp  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "SCAPP verification requires CuPy. Install with the cuda extra."
        ) from exc

    sample_rate_hz = 625e6
    full_scale_mv = 1000.0
    q = _build_quantised(sample_rate_hz)
    physical_setup = AWGPhysicalSetupInfo(logical_to_hardware_map={"H": 0, "V": 1})
    n_channels = int(physical_setup.N_ch)
    segment_lengths = tuple(int(seg.n_samples) for seg in q.segments)

    with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
        card.stop()
        card.card_mode(spcm.SPC_REP_STD_SEQUENCE)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(full_scale_mv * units.mV)
        channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(sample_rate_hz * units.Hz)
        clock.clock_output(False)

        full_scale = int(card.max_sample_value()) - 1

        # 1) CPU upload + CPU readback
        repo_cpu = QIRtoSamplesSegmentCompiler(
            quantised=q,
            physical_setup=physical_setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        ).compile_to_card_int16(gpu=False, output="numpy")
        upload_sequence_program(repo_cpu, mode="cpu", card=card, upload_steps=True)
        rb_cpu = _to_segment_dict(
            readback_sequence_segments_to_numpy(
                card=card,
                n_channels=n_channels,
                segment_lengths=segment_lengths,
            )
        )

        # 2) SCAPP upload + CPU readback
        repo_gpu = QIRtoSamplesSegmentCompiler(
            quantised=q,
            physical_setup=physical_setup,
            full_scale_mv=full_scale_mv,
            full_scale=full_scale,
        ).compile_to_card_int16(gpu=True, output="cupy")
        upload_sequence_program(repo_gpu, mode="scapp", card=card, upload_steps=True)
        rb_scapp = _to_segment_dict(
            readback_sequence_segments_to_numpy(
                card=card,
                n_channels=n_channels,
                segment_lengths=segment_lengths,
            )
        )

        # Compare CPU-upload vs SCAPP-upload card memory.
        worst = 0
        for idx in range(len(segment_lengths)):
            a = rb_cpu[idx]
            b = rb_scapp[idx]
            diff = np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32)
            max_abs = int(np.max(np.abs(diff)))
            worst = max(worst, max_abs)
            print(f"segment {idx}: shape={a.shape} max_abs_diff={max_abs}")

        if worst == 0:
            print("PASS: CPU upload and SCAPP upload read back identically.")
        else:
            print(f"WARNING: max absolute difference is {worst} LSB.")


if __name__ == "__main__":
    main()

