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
    AWGProgramBuilder(sample_rate=fs)
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
    .build_ir()
)
print(ir.duration_s, "seconds")
```

### 2) Compile to per-segment int16 samples

```python
from awgsegmentfactory import compile_sequence_program

compiled = compile_sequence_program(
    ir,
    logical_channel_to_hardware_channel={"H": 0, "V": 1},
    gain=1.0,
    clip=0.9,
    full_scale=32767,
)

print("segments:", len(compiled.segments))
print("steps:", len(compiled.steps))
```

### 3) Debug helpers (optional)

Debug helpers live in `awgsegmentfactory.debug` and require the `dev` dependency group
(matplotlib / ipywidgets).

- Grid/timeline debug (Jupyter): see `examples/debugging.py`
- Sample-level debug with segment boundaries: see `examples/sequence_samples_debug.py`

## Compilation stages (mental model)

1) **Build** (`src/awgsegmentfactory/builder.py`)
   - `AWGProgramBuilder` records your fluent calls into a spec.
2) **Intent IR** (`src/awgsegmentfactory/ir.py`)
   - `ProgramSpec` contains logical channels/definitions/segments and a list of ops per segment.
3) **Compile IR** (`src/awgsegmentfactory/program_ir.py`)
   - `ProgramIR` groups the program into segments and “parts”, where each part has an
     integer length `n_samples` and a per-logical-channel primitive: `(start, end, interp, tau)`.
   - Produced by `resolve_program_ir()` in `src/awgsegmentfactory/resolve.py`.
4) **Quantise for hardware** (`src/awgsegmentfactory/sequence_compile.py`)
   - `quantize_program_ir()` adjusts segment lengths to hardware constraints:
     step size, minimum segment size, and (for loopable constant segments) wrap-continuous
     frequency snapping for the final segment length.
5) **Samples** (`src/awgsegmentfactory/sample_compile.py`)
   - `compile_sequence_program()` synthesises per-segment int16 waveforms plus a sequence
     step table (`CompiledSequenceProgram`).

For plotting/state queries there is also a debug view:
- `ResolvedTimeline` (`src/awgsegmentfactory/timeline.py`) and `ProgramIR.to_timeline()`

## Structure understanding

'IR' means 'Intermediate Representation'.

We call the whole thing we are building up a 'Program'.

The builder defines a nice fluent builder API, that produces an IR formed of dataclasses. We call this product a 'Specification', that is, what you *want* to happen. It might not necessarily be buildable to real hardware, and it doesn't account for hardware details like sample rates, memory limits, etc... It does however need to contain the (metadata at this point) information about what future stages should do about hardware implementation that has an intentful decision that effects how it would operate in an experiment. It also has to be at the level that the points moving around we're describing have some linear position/frequency  relation, a phase associated with them, and an amplitude associated with them. In this code, this is the `ProgramSpec` class. This is primarily formed of `SegmentSpec`s which are formed of 'operations' (`Op`s). 

This is then turned into a more verbose, but still logical, implementation plan at this point, some hardware considerations have been made. That is the 'program_ir' (the IR which is good for compilation) we have:
A whole program is now represented by:
`ProgramIR`, which is formed of `SegmentIR`s that themselves are composed of `PartIR`s, which are composed of `LogicalChannelPartIR` (what each channel does in this part).

The transformation of a `ProgramSpec` to a `ProgramIR` is done by the `resolve_program_ir()`.

(to be honest, this is a quite awkward middle ground point...)

This is then further refined to hardware implementation by quantising everything to the quanta dictated by sensible hardware implementation. This is done by a function `quantize_program_ir()` which maps: `ProgramIR` -> `ProgramIR` and also outputs some quantisation metadata (`SegmentQuantizationInfo`). At this point we have an IR that is primed to produce AWG Samples to go into the hardware's RAM.

Interpretation of the `ProgramIR` to produce samples is done by `compile_sequence_program()` which maps `ProgramIR` and some hardware considerations to `CompiledSequenceProgram`. This contains the 'segments' and 'steps' information in the form of lists of `CompiledSegment` and `SequenceStep` respectively. This maps directly to the nature if the hardware where `[CompiledSegment]` is the data memory, and `[SequenceStep]` is the sequence memory.


## Repository guide (reading order)

1) `examples/compilation_stages.py` – end-to-end overview of the pipeline.
2) `src/awgsegmentfactory/builder.py` – fluent API and spec construction.
3) `src/awgsegmentfactory/ir.py` – intent IR (ops/spec types).
4) `src/awgsegmentfactory/resolve.py` – resolver (`ProgramSpec` → `ProgramIR`).
5) `src/awgsegmentfactory/program_ir.py` – compile IR dataclasses and helpers.
6) `src/awgsegmentfactory/sequence_compile.py` – segment quantisation + wrap snapping.
7) `src/awgsegmentfactory/sample_compile.py` – synthesis into int16 samples + steps.
8) `src/awgsegmentfactory/timeline.py` – debug timeline spans and interpolation.
9) `src/awgsegmentfactory/debug/` – optional plotting helpers (Jupyter + matplotlib).

## Notes / current limitations

- `phases="auto"` is currently a placeholder (phases default to 0).
- `warn_df` is recorded on `hold()` but not currently used to emit warnings during snapping.
- Calibration objects are stored on `ProgramSpec.calibrations` but not yet consumed by ops.
