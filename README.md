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
)

compiled = compile_sequence_program(
    quantized,
    gain=1.0,
    clip=0.9,
    full_scale=32767,
    logical_channel_to_hardware_channel={"H": 0, "V": 1},
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

If you pass an `OpticalPowerToRFAmpCalib` calibration object to `compile_sequence_program(...)`, then `amps` in the
IR are treated as *desired optical power* (arbitrary units), and sample synthesis converts `(freq, optical_power)`
to the RF synthesis amplitudes actually used for sample generation.

### Calibration concepts

`AODSin2Calib` models a single hardware channel with:

`optical_power(freq, rf_amp_mV) ≈ g(freq) * sin^2((π/2) * rf_amp_mV / v0(freq))`

- `g(freq)`: maximum reachable optical power (arb) at each frequency.
- `v0(freq)`: RF-amplitude scale (mV) controlling where saturation occurs.
- Inversion is used at compile time:
  desired optical power -> required RF amplitude.
- Behavior at limits:
  - negative optical powers are clamped to `0`.
  - requests above reachable power are clamped just below full scale of the model (`1 - y_eps` in normalized space).
  - frequencies outside calibrated bounds are evaluated using edge-clamped normalized frequency.

`AODSin2Calib` also stores:
- `freq_min_hz` and `freq_max_hz`: where data supports the fit.
- `traceability_string`: free-form provenance (filename, lab-book note, date, etc).
- derived `best_freq_hz`: frequency with highest fitted `g(freq)` within `[freq_min_hz, freq_max_hz]`.

### Calibration objects

Built-in calibration objects (`src/awgsegmentfactory/calibration.py`):

- `AODSin2Calib`
  - single-channel model + metadata.
  - serializable via `serialise` / `deserialise`.

- `AWGCalibration`
  - container passed to `compile_sequence_program(..., optical_power_calib=...)`.
  - serializable via `serialise` / `deserialise` and `to_file` / `from_file`.
  - fields:
    - `N_ch: int`
    - `logical_to_hardware_map: Dict[str, int]`
    - `channel_calibrations: Tuple[AODSin2Calib, ...]`
  - routes each logical channel in the IR to the correct physical-channel calibration.

### Practical workflow

1) Measure calibration data from your setup:
   - DE-compensation JSON (`DE_RF_calibration`),
   - `.awgde`,
   - or CSV point cloud `(freq_MHz, rf_amp_mV, power_arb)`.
2) Fit calibrations with:
   - `python -m awgsegmentfactory.tools.fit_optical_power_calibration ...`
   - pass one `--input-data-file` per hardware channel.
   - optionally set mapping with repeated `--logical-to-hardware-map`, e.g. `H=0`, `V=1`.
   - optionally set provenance with repeated `--traceability-string`.
3) Save JSON output (`--write-out`) and load as `AWGCalibration.from_file(...)`.
4) Compile with calibration:
   - `compile_sequence_program(..., optical_power_calib=awg_calibration)`.

Examples:
- `examples/optical_power_calibration_demo.py` (toy sin² model, with/without calibration overlay)
- `examples/fit_optical_power_calibration.py` (fit and inspect generated `AODSin2Calib`/`AWGCalibration`)
- `examples/sequence_samples_debug_sin2_calib.py` (fit sin² from file, compile with calibration, and debug samples)

## Compilation stages (mental model)

```mermaid
flowchart LR
    B[AWGProgramBuilder]
    I[IntentIR]
    R[ResolvedIR]
    Q[QuantizedIR]
    C[CompiledSequenceProgram]

    B -- build_intent_ir --> I
    I -- resolve_intent_ir --> R
    R -- quantize_resolved_ir --> Q
    Q -- compile_sequence_program --> C

    CAL[Calibrations: AODSin2Calib / AWGCalibration] --> C

    R -- to_timeline --> TL[ResolvedTimeline]
    Q -- debug --> DBG[sequence_samples_debug]
```

1) **Build (intent)** (`src/awgsegmentfactory/builder.py`)
   - `AWGProgramBuilder` records your fluent calls into an `IntentIR` (`build_intent_ir()`).
2) **Intent IR** (`src/awgsegmentfactory/intent_ir.py`)
   - `IntentIR` is continuous-time intent: logical channels/definitions/segments and ops with `time_s` in seconds.
3) **Resolve (discretize)** (`src/awgsegmentfactory/resolve.py` + `src/awgsegmentfactory/resolved_ir.py`)
   - `resolve_intent_ir(intent, sample_rate_hz=...)` converts seconds → integer `n_samples` and produces `ResolvedIR`.
4) **Quantise for hardware** (`src/awgsegmentfactory/quantize.py`)
   - `quantize_resolved_ir(resolved)` returns a `QuantizedIR`:
     a quantized `ResolvedIR` plus `SegmentQuantizationInfo`.
5) **Samples** (`src/awgsegmentfactory/synth_samples.py`)
   - `compile_sequence_program(quantized, logical_channel_to_hardware_channel=..., ...)` synthesises per-segment int16 waveforms plus a sequence
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
10) `src/awgsegmentfactory/optical_power_calibration_fit.py` – fitting helpers for `AODSin2Calib`.
11) `src/awgsegmentfactory/debug/` – optional plotting helpers (Jupyter + matplotlib).

## Notes / current limitations

- `phases="auto"` currently means phases default to 0; use per-segment `phase_mode` for
  crest-optimised/continued phases during compilation.
- `OpticalPowerToRFAmpCalib` calibrations (e.g. `AODSin2Calib`) are consumed during
  `compile_sequence_program(..., optical_power_calib=...)` to convert `(freq, optical_power)` → RF synthesis amplitudes.

## Roadmap

See `TODO.md`.
