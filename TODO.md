# TODO / Roadmap

This file tracks higher-level features we’d like to implement in AWGSegmentFactory.
It replaces `plan.md`, which had a mixture of historical design notes and completed work.

## ndscan integration (requested)

- [ ] Add an `ndscan` “driver fragment” skeleton (e.g. `AWGProgrammerFragment`)
  - Host side: `prepare_point()` builds/resolves/quantizes/compiles and uploads only changed segments
  - Kernel side: `device_setup()` / `run_once()` issues RTIO triggers to advance/start playback
- [ ] Define a clean way for fragments to contribute to a shared `AWGProgramBuilder`
  - Depth-first fragment tree order determines segment insertion order
  - Avoid “global AWG state”; fragments append intent to the shared builder
- [ ] Implement per-segment hashing + caching across scan points
  - Detect unchanged segments and skip recompilation / re-upload
  - Keep “segment count changed” as the slow path
- [ ] Document the recommended integration pattern (minimal example experiment)

## Hardware upload (Spectrum)

- [ ] Implement `upload_sequence_program(..., mode="cpu")` for Spectrum sequence-mode DMA
  - Provide a stable API surface (even if the backend evolves)
  - Keep `examples/spcm/` as the “working reference” while this matures
- [ ] (Optional) Implement `mode="scapp"` for GPU-resident buffers when compiling with `gpu=True, output="cupy"`

## Compiler pipeline improvements

- [ ] Add small, explicit “hash inputs” helpers for caching (Resolved/Quantized/Compiled)
- [ ] (Optional) Parallelize segment compilation on CPU (threads/processes) for large programs
- [ ] Tighten and document endpoint conventions (e.g. chirps/phase integration) where needed
- [ ] Reduce allocations in synthesis (especially GPU)
  - Preallocate per-segment output buffers instead of list + `concatenate`
  - Special-case pure holds to avoid allocating full `(n_samples, n_tones)` arrays
- [ ] Make crest-factor optimisation more configurable/faster
  - Expose tuning knobs (passes/grid selection)
  - Consider caching per `(freqs, amps)` signature when many segments repeat

## Calibration + modeling

- [ ] Integrate `PositionToFreqCalib` into higher-level builder ops (e.g. `move_dx_um(...)`)
- [ ] Expand `AODTanh2Calib` fit tooling for lab workflows
  - Robust fitting (outlier handling / weighting)
  - Better validation plots and fit-quality checks
  - Import/export helpers for storing “CONST-like” calibration blobs per AOD/channel

## Code health / maintainability

- [ ] Move `LogicalChannelState` out of `src/awgsegmentfactory/resolved_timeline.py` so the “timeline” module can stay debug-only
- [ ] Refactor `resolve_intent_ir` op handling into a dispatch table (similar to `_INTERP_BY_KIND` in interpolation)
- [ ] Avoid debug modules importing private compiler helpers (e.g. promote `_interp_logical_channel_part` to a shared internal module)
- [ ] Improve typing for NumPy/CuPy compatibility (use `Any`/Protocol/ArrayLike where appropriate)
- [ ] Make the upload API’s stability explicit (mark as experimental or move under an experimental namespace until implemented)
- [ ] Add unit tests for interpolation edge cases (`geo_ramp` validation, `exp` including τ→0 behavior)

## Debug / UX

- [ ] Improve debug plots for large tone counts (sampling/decimation, tone coloring, etc.)
- [ ] Add small CLI wrappers for common calibration/debug tasks (optional)
