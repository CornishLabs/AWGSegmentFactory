# AWG Segment Factory

A small library for building **AWG sequence-mode programs** from a fluent, stateful
builder API, then compiling them into **per-segment int16 samples** and a **sequence
step table** (e.g. for Spectrum sequence replay mode).

The design goal is to keep a clear separation between:
- **User intent** (what you asked for)
- **Compiler-friendly IR** (explicit, integer-sample primitives)
- **Hardware constraints** (segment quantisation, wrap-continuous holds, minimum sizes)
- **Samples** (what you upload to the card)

## To run

This package uses `uv` to manage dependencies. To run any of the examples:
```
uv run examples/recreate_current.py
```

Python requirement: `>=3.13` (see `pyproject.toml`).

### Restricted environments (optional)

If your environment blocks access to `~/.cache` (e.g. some sandboxes/CI), run uv with a repo-local cache:

```
uv run --cache-dir .uv-cache examples/recreate_current.py
```

`uv` manages a virtualenv in `.venv`. Install/update deps with:
```
uv sync --dev
```

## Quick start

### 1) Build a program (builder → IR)

```python
import numpy as np
from awgsegmentfactory import AWGProgramBuilder

fs = 625e6

ir = (
    AWGProgramBuilder()
    .logical_channel("H")
    .logical_channel("V")
    .define("init_H", logical_channel="H", freqs=[90e6], amps=[0.3], phases="auto")
    .define("init_V", logical_channel="V", freqs=[100e6], amps=[0.3], phases="auto")
    .segment("wait", mode="wait_trig")     # loops until trigger
        .tones("H").use_def("init_H")
        .tones("V").use_def("init_V")
        .hold(time=200e-6)
    .segment("chirp_H", mode="once")       # one-shot segment
        .tones("H").move(df=+2e6, time=50e-6, idxs=[0])
    .build_resolved_ir(sample_rate_hz=fs)
)
print(ir.duration_s, "seconds")
```

### 2) Compile to per-segment int16 samples

```python
from awgsegmentfactory import compile_sequence_program, quantize_resolved_ir

quantized = quantize_resolved_ir(
    ir,
    logical_channel_to_hardware_channel={"H": 0, "V": 1},
)

compiled = compile_sequence_program(
    quantized,
    gain=1.0,
    clip=0.9,
    full_scale=32767,
)

print("segments:", len(compiled.segments))
print("steps:", len(compiled.steps))
```

### GPU synthesis (optional)

If you have CuPy + an NVIDIA GPU available, `compile_sequence_program(..., gpu=True)` runs the
sample-synthesis stage on the GPU (resolve/quantize are still CPU).

- `output="numpy"` (default): returns NumPy int16 buffers (GPU→CPU transfer once per segment).
- `output="cupy"`: keeps int16 buffers on the GPU (useful for future RDMA workflows).
  - To convert back to NumPy, use `compiled_sequence_program_to_numpy(...)`.

See `examples/benchmark_pipeline.py --gpu`.

### Phase modes (optional)

Each segment can set `phase_mode` to control how the *start phases* are chosen:

- `manual`: use the phases stored in the IR (from `.define(..., phases=[...])`, `.add_tone(phase=...)`, etc.).
- `optimise`: choose start phases to reduce crest factor based on the segment's start freqs/amps.
- `continue`: continue phases across segment boundaries for tones whose frequencies match, and optimise
  any new/unmatched tones while keeping continued tones fixed.

Notes:
- `phase_mode` is applied during `compile_sequence_program(...)` (sample synthesis). The debug
  timeline `ResolvedIR.to_timeline()` shows pre-optimised phases.
- `.define(..., phases="auto")` currently means "all zeros"; this is typically fine when using
  `phase_mode="optimise"`/`"continue"`.

### 3) Debug helpers (optional)

Debug helpers live in `awgsegmentfactory.debug` and require the `dev` dependency group
(matplotlib / ipywidgets).

- Grid/timeline debug (Jupyter): see `examples/debugging.py`
- Sample-level debug with segment boundaries (and optional 2D spot grid): see `examples/sequence_samples_debug.py`

## Hardware upload (Spectrum) (optional)

This repo includes working Spectrum examples under `examples/spcm/` (sequence mode, triggers, etc).
The library function `upload_sequence_program(...)` is a placeholder for a future stable API; today it raises
`NotImplementedError` for CPU upload and points at `examples/spcm/6_awgsegmentfactory_sequence_upload.py`.

## Optical-power calibration (optional)

If you attach an `OpticalPowerToRFAmpCalib` calibration object to the builder, then `amps` in the IR are treated
as *desired optical power* (arbitrary units), and `compile_sequence_program(...)` converts `(freq, optical_power)`
to the RF synthesis amplitudes actually used for sample generation.

Built-in calibrations (see `src/awgsegmentfactory/calibration.py`):
- `AODDECalib`: `optical_power ≈ DE(freq) * rf_amp^2` (simple, no saturation).
- `AODTanh2Calib`: `optical_power ≈ g(freq) * tanh^2(rf_amp / v0(freq))` (smooth saturation, globally invertible).

Examples:
- `examples/optical_power_calibration_demo.py` (toy DE model, with/without calibration overlay)
- `examples/fit_optical_power_calibration.py` (fit `AODTanh2Calib` from a DE-compensation JSON file and print a Python constant)
- `examples/sequence_samples_debug_tanh2_calib.py` (fit tanh² from file, attach to builder, and debug compiled samples)

## Compilation stages (mental model)

```mermaid
flowchart LR
    B[AWGProgramBuilder<br/>fluent API]
    I[IntentIR<br/>continuous-time ops]
    R[ResolvedIR<br/>integer-sample parts]
    Q[QuantizedIR<br/>hardware-aligned segments]
    C[CompiledSequenceProgram<br/>int16 segments + step table]

    B -->|build_intent_ir| I
    I -->|resolve_intent_ir| R
    R -->|quantize_resolved_ir| Q
    Q -->|compile_sequence_program| C

    B -->|attach (optional)| CAL[calibrations<br/>(e.g. AODTanh2Calib)]
    CAL -->|used during synthesis| C

    R --> TL[ResolvedTimeline<br/>(debug)]
    Q --> DBG[debug plots<br/>(sequence_samples_debug)]
```

1) **Build (intent)** (`src/awgsegmentfactory/builder.py`)
   - `AWGProgramBuilder` records your fluent calls into an `IntentIR` (`build_intent_ir()`).
2) **Intent IR** (`src/awgsegmentfactory/intent_ir.py`)
   - `IntentIR` is continuous-time intent: logical channels/definitions/segments and ops with `time_s` in seconds.
3) **Resolve (discretize)** (`src/awgsegmentfactory/resolve.py` + `src/awgsegmentfactory/resolved_ir.py`)
   - `resolve_intent_ir(intent, sample_rate_hz=...)` converts seconds → integer `n_samples` and produces `ResolvedIR`.
4) **Quantise for hardware** (`src/awgsegmentfactory/quantize.py`)
   - `quantize_resolved_ir(resolved, logical_channel_to_hardware_channel=...)` returns a `QuantizedIR`:
     a quantized `ResolvedIR` plus `SegmentQuantizationInfo`.
5) **Samples** (`src/awgsegmentfactory/synth_samples.py`)
   - `compile_sequence_program(quantized, ...)` synthesises per-segment int16 waveforms plus a sequence
     step table (`CompiledSequenceProgram`).

For plotting/state queries there is also a debug view:
- `ResolvedTimeline` (`src/awgsegmentfactory/resolved_timeline.py`) and `ResolvedIR.to_timeline()`

## IR terminology

- **Intent IR**: continuous-time spec in seconds; “what you want”.
- **Resolved IR**: sample-quantized primitives (per-part integer sample counts); “what you mean”.
- **Quantized IR**: hardware-aligned segment lengths + optional wrap snapping; “what you can upload”.
- **Compiled program**: final int16 segment buffers + step table; “what the card plays”.

## Repository guide (reading order)

1) `examples/compilation_stages.py` – end-to-end overview of the pipeline.
2) `src/awgsegmentfactory/builder.py` – fluent API and spec construction.
3) `src/awgsegmentfactory/intent_ir.py` – intent IR (ops/spec types).
4) `src/awgsegmentfactory/resolve.py` – resolver (`IntentIR` → `ResolvedIR`).
5) `src/awgsegmentfactory/resolved_ir.py` – resolved IR dataclasses and helpers.
6) `src/awgsegmentfactory/quantize.py` – quantisation (`ResolvedIR` → `QuantizedIR`) + wrap snapping.
7) `src/awgsegmentfactory/synth_samples.py` – synthesis (`QuantizedIR` → `CompiledSequenceProgram`).
8) `src/awgsegmentfactory/resolved_timeline.py` – debug timeline spans and interpolation.
9) `src/awgsegmentfactory/calibration.py` – calibration interfaces and built-in models.
10) `src/awgsegmentfactory/optical_power_calibration_fit.py` – fitting helpers for `AODTanh2Calib`.
11) `src/awgsegmentfactory/debug/` – optional plotting helpers (Jupyter + matplotlib).

## Notes / current limitations

- `phases="auto"` currently means phases default to 0; use per-segment `phase_mode` for
  crest-optimised/continued phases during compilation.
- `OpticalPowerToRFAmpCalib` calibrations (e.g. `AODDECalib` / `AODTanh2Calib`) are consumed during
  `compile_sequence_program(...)` to convert `(freq, optical_power)` → RF synthesis amplitudes. Other
  calibration types are currently stored on `IntentIR.calibrations` for future higher-level ops.
