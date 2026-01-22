# For Humans & AI

Here’s the intent of each file (as we’ve been shaping it), plus a good reading order for someone new.

## Read order for a human

1. **`examples/...`**

   * Start here. Shows the *intended* fluent API and what kinds of sequences you can express.
2. **`src/awgsegmentfactory/builder.py`**

   * The “front-end”: the fluent chain that records user intent into a spec.
3. **`src/awgsegmentfactory/ir.py`**

   * The “data model”: the classes that represent program specs + resolved IR.
4. **`src/awgsegmentfactory/resolve.py`** (or whatever your resolver file is named)

   * The “compiler stage 1”: turns the spec into resolved, explicit per-part start/end states.
5. **`src/awgsegmentfactory/debug/plot.py`** (or `src/awgsegmentfactory/debug/__init__.py`)

   * Visualization: takes resolved IR and produces slider plots / animations.
6. **`src/awgsegmentfactory/__init__.py`**

   * Public exports only (what users import).

---

## What each file does

### `builder.py`

**Intent:** ergonomic API.
**Does:** collects user calls (`segment()`, `tones()`, `move()`, `hold()`, etc.) into an **IntentIR** (a list of segments + ops).
**Should NOT do:** heavy computation, snapping, interpolation math, sampling.

### `ir.py`

**Intent:** formal, stable “contract” between stages.
**Does:** defines:

* **Spec-level types** (what the builder emits): segments, ops (move/ramp/hold/remap), definitions, logical channels
* **Resolved IR types** (what the resolver emits): a timeline broken into parts with explicit `start_state`, `end_state`, `duration`, `interp`, etc.

### `resolve.py` (or similar)

**Intent:** “compiler pass” that makes everything explicit and independent.
**Does:** walks segments in order, maintains a persistent **current state**, applies ops, and outputs:

* segment start/end state
* parts (each part is a tween from start→end over a duration)
* frequency snapping for `wait_trig` holds
* duration rounding to ≥ 1 sample

This is where the “state should carry across segments” rule is enforced.

### `debug/plot.py`

**Intent:** rapid sanity checking.
**Does:** renders the resolved IR as a time-slider animation:

* maps H freq→x, V freq→y
* maps amp→marker size/alpha
* fixes axis limits across the whole program so motion is visually stable

### `__init__.py`

**Intent:** clean public API surface.
**Does:** re-exports `AWGProgramBuilder`, calibration objects, and the main IR/stage functions.

---

## Quick mental model

* **Builder** = *nice to write*
* **Spec (in `ir.py`)** = *recorded intent*
* **Resolver** = *turn intent into explicit timeline states*
* **Resolved IR** = *easy to compile & cache*
* **Debug plot** = *see if it matches your intent*

---

# For AI

## What we’re building

### Goal

A standalone, composable **AWG program “factory”** that lets you write readable, high-level timeline code like:

```python
prog = (
  AWGProgramBuilder()
    .logical_channel("H").logical_channel("V")
    .define("loading_H", logical_channel="H", freqs=[...], amps=[...], phases="auto")
    ...
    .segment("move_row_up", mode="loop_n", loop=1)
      .tones("V").move(df=2e6, time=1.0, idxs=[0])
    .segment("wait_for_trigger_B", mode="wait_trig")
      .hold(time=1.0)
    .build_resolved_ir(sample_rate_hz=625e6)
)
```

and produces a **resolved intermediate representation (IR)** that is “compiler-friendly”: each segment becomes a list of **independent parts** where each part explicitly has:

* start freqs/amps/phases per logical channel
* end freqs/amps/phases per logical channel
* duration
* interpolation (linear/exp/min-jerk/hold)
* plus metadata for wait-trig segments (snapping / quantisation)

This IR is then (later) compiled into raw AWG samples / segments for Spectrum `spcm` (sequence mode, loop_n, wait_trig, hotswap DMA, etc.). We’re not implementing the final sample compiler yet; we’re building the IR and debug tooling first.

---

## Core design decisions we agreed

### 1) Builder declares a continuous timeline (stateful cursor)

* Ops with `time=0` are allowed and **change state without advancing time**.
* **Segments must be at least 1 sample long** (duration rounds up to ≥ 1 sample). A “sync” segment is the minimal 1-sample hold.
* Segment length is the sum of durations of timed ops in it; if the segment contains only time=0 ops → error (no duration).

### 2) Segment boundaries preserve state (no implicit resets)

* The resolved program must **carry end state → next segment start state**.
* Definitions (`define(...)`) are templates; they only affect the active state when explicitly “applied” (e.g. `use_def`, `remap_from_def`).
* We discovered and fixed a bug where logical-channel state was rebuilt from definitions at segment boundaries, causing moves to “reset”.

### 3) Quantised hold semantics

* For `mode="wait_trig"` segments: user supplies `hold(time=...)`.
* During resolving/compiling: frequencies are **snapped/rounded** to the nearest values that make the waveform wrap-continuously for that hold length (quantisation grid depends on `sample_rate` and segment length in samples).
* We default to rounding; no extra modes exposed yet.
* `warn_df` lets you warn if snapping delta exceeds some threshold.

### 4) Unified tone operations

We aim for consistent, minimal primitives:

* `tones("H").use_def("loading_H")` applies a definition (absolute state set).
* `tones("V").move(df=..., time=..., idxs=[...])` shifts selected tones.
* `tones("H").ramp_amp_to(amps=..., time=..., kind="exp", tau=...)`
* `tones("H").off(idxs=...)` means amplitude→0 but tone identity retained.
* `tones("H").remove(idxs=...)` removes tones (indexing changes; builder makes this explicit).
* `tones("H").remap_from_def(target_def=..., src=[...], dst="all", time=..., kind="min_jerk")` for hotswap/rearrange.

We’re *not* implementing full “grid semantics” (Cartesian NxM) yet; we treat H and V as independent logical channels. Later we can add grid metadata / helpers without changing core IR.

### 5) Debug-first workflow

Before sample compilation, we generate **debug plots** driven by IR:

* A 2D scatter (x,y) where:

  * x ∝ H frequency
  * y ∝ V frequency
  * marker size or alpha ∝ amplitude (or product of amplitudes)
* A time slider (Jupyter ipywidgets) to scrub frames.
* Fixed axis limits (computed from full program extents) so points visibly move.

This lets you iterate quickly in a notebook: edit builder code → run cell → see motion.

---

## What the IR looks like (conceptually)

Resolved IR is “flat and explicit”, no dependence on builder chain:

* `ResolvedIR(sample_rate_hz, segments=[ResolvedSegment...])`
* `ResolvedSegment(name, mode, loop, parts=[ResolvedPart...], start_state, end_state)`
* `ResolvedPart(duration_s, interp_kind, start_state, end_state, metadata...)`
* `State` contains per logical channel:

  * list of tones: freqs_hz[], amps[], phases_rad[]
  * optional tone ids for tracking/remove/remap

Importantly:

* Each part is self-contained: “compute these tones from start to end over duration with interp X”.
* That makes compilation parallelisable and cacheable (hash each segment/part).

---

## Plan for compilation and caching (later)

We expect AWG upload is expensive, so compilation should be incremental:

* Hash the **resolved segment** (or per-part) based on:

  * durations (in samples)
  * snapped freqs
  * amps/phases
  * interp type/params
* If a scan only affects parts 3–4, only those change hash → only those are recomputed and re-uploaded.
* Segment count changing is rare and OK to be expensive.

---

## How this fits into ndscan (no ndscan hacks)

### The idea

Treat the AWG program builder as a **factory object** that lives inside a fragment tree, and have a minimal ndscan fragment (“NDSP driver”) that:

1. builds/resolves the program per point (only if needed),
2. uploads changed segments to the hardware on the host,
3. triggers playback from RTIO at the right times.

### Roles

#### A) Factory / builder (pure Python)

* `AWGProgramBuilder` and calibration transforms are **hardware-agnostic**.
* It consumes “physics-ish” ops (move in µm, etc.) via calibration objects and emits resolved IR.

This makes it usable:

* standalone (unit tests, notebooks)
* inside ndscan

#### B) ndscan fragment: “AWGProgrammerFragment” (host-side)

* Owns:

  * AWG connection / handle
  * last uploaded hashes
  * current uploaded program id / segment ids
* During ndscan point execution it does:

  * `prepare_point()` (host): resolve builder → IR, snap holds, compute segment hashes, upload only changed segments, set up sequence tables.
  * `device_setup()` or `run_once()` (kernel): arrange RTIO trigger events (TTL pulses) to advance segments / start playback.
  * `device_cleanup()` (kernel): safe state if needed.

Key: uploading is not RTIO-safe; it’s host work. Triggering is RTIO.

#### C) Passing the factory object through fragments (composability)

Fragments don’t share “global AWG state”; they *contribute* to the factory:

* Top fragment creates builder/context object once.
* Subfragments add segments/ops in **depth-first fragment tree order** (your requirement).
* This naturally determines segment insertion order without manual indexing.

Mechanically in ndscan:

* In `build_fragment()`, fragments declare they “use AWG” and get a reference to a shared builder (passed down or stored on parent).
* Each fragment has a method like `emit_awg(self, b: AWGProgramBuilder)` that appends its piece.
* The experiment’s `host_setup` / `prepare_point` assembles the full builder by walking fragments DFS.

### Where scanning fits

* If a parameter is scanned that affects AWG, it’s handled in `prepare_point()`:

  * builder gets current `.get()` values from ndscan param handles
  * only affected segments change hash → only those segments recompiled/uploaded

No need to put everything in `host_setup()`; `prepare_point` is the correct hook for per-point updates.

---

## Immediate next tasks (what Codex should implement)

1. **Fix/ensure state propagation** across segments (no resets unless `use_def` is called).
2. IR structures:

   * `ir.py`: dataclasses for `IntentIR`, `IntentSegment`, ops; and `ResolvedIR`, `ResolvedSegment`, `ResolvedPart`, `State`.
3. Resolver:

   * `resolve.py`: apply ops into a persistent `current_state` across segments.
   * implement hold snapping for wait_trig segments.
4. Debug tools:

   * `debug/plot.py`: produce fixed-axis 2D scatter + slider from `ResolvedIR`.
5. Notebook workflow:

   * example notebook cell: build program → `ir = prog.ir` → `show_debug(ir, ...)`

---

If you want, paste this into Codex along with your current file tree and ask it to:

* implement the resolver state model correctly,
* add fixed-axis limits to the plot helper,
* and scaffold the ndscan “driver fragment” interface (`prepare_point`, upload caching, RTIO trigger API) as a design stub (no hardware yet).
