# Phase 5.2 — Prepared Metal Model acceptance

Date: 2026-10-12. Scope: static FP32 rank <= 4 models using the Phase 5.1 Metal
operator subset. Public API: `MetalExecutionPlan` / `MetalExecutionContext` in
`tinyinfer/backend/metal/execution_plan.h`, linked with `TinyInfer::metal`.

## Delivered contracts

- Preparation owns a deep Model snapshot, topological order and MemoryPlan.
  Needed constants upload exactly once; contexts share immutable constants and
  five pipelines. Caller changes to the original Model cannot alter the plan.
- Each context allocates separate input buffers and one activation arena.
  Intermediate values remain on device. No logical device allocation or upload
  occurs during repeated runs without rebinding. Host transfer staging remains
  an allocation per bind/output; this is not an allocation-free inference claim.
- Before overwriting a reused slot, execution waits the completion prefix through
  its previous occupant's final consumer. Output slots remain pinned. `run()`
  waits all operations before success; failures propagate without CPU retry.
- Binding validates IDs, shape and dtype. By-value Tensor lvalues deep-copy to
  contiguous storage; moved views retain layout. Contiguous view offsets are
  respected during staging. Successful rebinding invalidates previous output.
- `output()` accepts declared outputs only and returns an independent host Tensor.
  It requires a successful run since binding. No mutable Graph, device buffer,
  constant tensor or runtime accessor is exposed.
- Contexts are movable, noncopyable and retain the plan. Moved-from operations
  fail. Contexts sharing a queue require external serialization.
- Default Strict rejects unavailable hardware, unsupported models and moved
  noncontiguous inputs. Explicit WholeModelCpu selects Prepared CPU Auto for the
  entire model. A CPU backup is prepared eagerly under this policy (extra host
  snapshot/packing cost), including runtime layout fallback. Rebinding all inputs
  contiguously can return that context to Metal. Placement exposes the reason.
- Statistics distinguish preparation uploads, arena footprint, completed GPU/CPU
  runs, input/output payload and reuse waits. Device statistics are cumulative
  shared-runtime logical requests; they are not resident memory or timing.

## Verification

Mac: Intel Iris Plus Graphics 645, x86_64, macOS 15.8.1, AppleClang 17.
Real-device tests require host GPU access, `TINYINFER_METAL_REQUIRE_DEVICE=ON`
and `MTL_DEBUG_LAYER=1`; skipped tests do not count as acceptance.

The new test covers both `phase3_mlp_gemm.onnx` and `rl_actor_mlp_tanh.onnx`,
golden outputs and Prepared CPU comparisons; changing inputs, repeated runs,
interleaved contexts, upload/allocation counts, independent output copies,
slot reuse, model snapshot, context lifetime/moves, strict rejection, static
and runtime fallback, GPU reentry, unavailable ordinal, dynamic MatMul RHS,
constant/input-only outputs, offset views, empty outputs and rank boundary.

Final gates: Release Metal **46/46**, CPU-only **43/43**, Metal host
ASan/UBSan **3/3**, native CPU Auto reference comparison **1/1**; zero skips.
Both public model examples complete three GPU runs. Commands, counts and
retained logs are recorded in `phase5.2-verification.json`.
ASan/UBSan validate host ownership and bounds, not shader memory. Device-fault
injection and simultaneous shared-context execution are outside this gate.

## Next stage

Phase 5.3 measures preparation, binding, run, output copy and complete inference
separately against Prepared CPU Auto; small models may be slower on GPU. No GPU
speedup, CUDA readiness, dynamic shape, arbitrary ONNX, KV Cache or attention
support is claimed by Phase 5.2.

Example: `./build-phase51-metal/examples/tinyinfer_metal_model tests/fixtures/phase3_mlp_gemm.onnx`.
