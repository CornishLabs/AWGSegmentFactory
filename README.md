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

See `examples/benchmark_pipeline.py --gpu`.

### 3) Debug helpers (optional)

Debug helpers live in `awgsegmentfactory.debug` and require the `dev` dependency group
(matplotlib / ipywidgets).

- Grid/timeline debug (Jupyter): see `examples/debugging.py`
- Sample-level debug with segment boundaries: see `examples/sequence_samples_debug.py`

## Compilation stages (mental model)

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

## Structure understanding

'IR' means 'Intermediate Representation'.

We call the whole thing we are building up a 'Program'.

The builder defines a fluent API that produces an IR formed of dataclasses. We call this product **intent**
(what you *want* to happen). It might not necessarily be buildable to real hardware, and it doesn't account
for later hardware constraints like sequence quantisation and minimum segment sizes.

In this code, that is the `IntentIR` class. This is primarily formed of `IntentSegment`s which contain
operations (`Op`s) with durations in seconds.

This is then turned into a more verbose, but still logical, implementation plan at this point, some hardware considerations have been made. That is the resolved IR (the IR which is good for compilation) we have:
A whole program is now represented by:
`ResolvedIR`, which is formed of `ResolvedSegment`s that themselves are composed of `ResolvedPart`s, which are composed of `ResolvedLogicalChannelPart` (what each logical channel does in this part).

The transformation of an `IntentIR` to a `ResolvedIR` is done by `resolve_intent_ir(intent, sample_rate_hz=...)`.

(to be honest, this is a quite awkward middle ground point...)

This is then further refined to hardware implementation by quantising everything to the quanta dictated by sensible hardware implementation. This is done by `quantize_resolved_ir()`, which maps:
`ResolvedIR` -> `QuantizedIR` (and includes `SegmentQuantizationInfo`).

Interpretation of the quantized IR to produce samples is done by `compile_sequence_program()`, which maps `QuantizedIR` to `CompiledSequenceProgram`. This contains the 'segments' and 'steps' information in the form of lists of `CompiledSegment` and `SequenceStep` respectively. This maps directly to the nature of the hardware where `[CompiledSegment]` is the data memory, and `[SequenceStep]` is the sequence memory.


## Repository guide (reading order)

1) `examples/compilation_stages.py` – end-to-end overview of the pipeline.
2) `src/awgsegmentfactory/builder.py` – fluent API and spec construction.
3) `src/awgsegmentfactory/intent_ir.py` – intent IR (ops/spec types).
4) `src/awgsegmentfactory/resolve.py` – resolver (`IntentIR` → `ResolvedIR`).
5) `src/awgsegmentfactory/resolved_ir.py` – resolved IR dataclasses and helpers.
6) `src/awgsegmentfactory/quantize.py` – quantisation (`ResolvedIR` → `QuantizedIR`) + wrap snapping.
7) `src/awgsegmentfactory/synth_samples.py` – synthesis (`QuantizedIR` → `CompiledSequenceProgram`).
8) `src/awgsegmentfactory/resolved_timeline.py` – debug timeline spans and interpolation.
9) `src/awgsegmentfactory/debug/` – optional plotting helpers (Jupyter + matplotlib).

## Notes / current limitations

- `phases="auto"` is currently a placeholder (phases default to 0).
- Calibration objects are stored on `IntentIR.calibrations` but not yet consumed by ops.
