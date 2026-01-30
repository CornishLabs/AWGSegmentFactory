from awgsegmentfactory import AWGProgramBuilder, LinearPositionToFreqCalib
import numpy as np

from sequence_repo import recreate_mol_exp

fs = 625e6  # 625MHz
ir = (
    recreate_mol_exp()
    .build_resolved_ir(sample_rate_hz=fs)
)


print(
    f"segments: {len(ir.segments)} | duration: {ir.duration_s * 1e3:.3f} ms | fs={ir.sample_rate_hz / 1e6:.1f} MHz"
)
for seg in ir.segments:
    print(
        f"\n--- {seg.name} --- mode={seg.mode} loop={seg.loop} samples={seg.n_samples}"
    )
    if not seg.parts:
        continue
    for i, part in enumerate(seg.parts):
        lc_H = part.logical_channels.get("H")
        lc_V = part.logical_channels.get("V")
        h_desc = f"H:{lc_H.interp.kind}" if lc_H is not None else "H:—"
        v_desc = f"V:{lc_V.interp.kind}" if lc_V is not None else "V:—"
        print(
            f"part {i}: n={part.n_samples} ({part.n_samples / ir.sample_rate_hz * 1e6:.2f} µs) {h_desc} {v_desc}"
        )
