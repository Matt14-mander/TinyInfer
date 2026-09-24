# Graph optimizer architecture

Phase 4 begins with a model-aware graph pass pipeline. The optimizer does not
mutate a Graph in place: each pass receives a validated Model and returns a new
validated Model together with the ValueId mapping and one statistics record.

```text
Model
  |
  v
PassManager
  |
  +--> GraphPass A --> Model A + ValueId map + statistics
  |
  +--> GraphPass B --> Model B + ValueId map + statistics
  |
  v
OptimizationResult
  ├── optimized Model
  ├── original-to-final ValueId mapping
  └── ordered pass statistics
```

## GraphPass

`GraphPass` exposes a stable pass name and a `run(const Model&)` operation. A
pass works at the Model boundary because graph reconstruction may change
ValueIds while external input and output names must remain valid.

Each pass returns an `OptimizationResult`. When called directly, its
`value_mapping` maps the pass input Graph to the returned Model Graph, and its
statistics vector contains exactly one record.

## OptimizationResult

The result owns the optimized Model and records:

- a ValueId mapping from the input of the complete pipeline to the final Graph;
- `kInvalidValueId` for an internal value removed by a future pass;
- ordered `PassStatistics` with before/after node and value counts;
- the number of rewritten nodes and whether each pass changed the model.

`mapped_value()` performs checked lookup. `changed()` reports whether any pass
in the pipeline declared a transformation.

## PassManager

`PassManager` owns an ordered list of passes. After each pass it validates:

- the mapping covers every Value in the pass input Graph;
- every non-removed mapped ValueId exists in the returned Graph;
- external input and output counts, names, order, and mapped IDs are preserved;
- the statistics record matches the pass name and actual Graph sizes.

It then composes the per-pass mapping into one original-to-final mapping. An
empty manager returns an independent Model copy, an identity mapping, and no
statistics.

## NoOpPass

`NoOpPass` is the first pipeline validation pass. It returns an independent
copy of the Model, an identity ValueId mapping, unchanged Graph counts, zero
rewritten nodes, and `changed=false`.

Because TinyInfer Tensor copies have value semantics, copying the Model also
deep-copies constant Tensor buffers. NoOpPass is a correctness scaffold, not a
performance optimization. Future rebuilding passes can move newly constructed
constants into the result and Constant Folding can create only the constants
it needs.

The current test covers direct inference before and after the pass, a two-pass
pipeline, an empty pipeline, interface preservation, graph structure, constant
ownership, mapping lookup, and statistics.

## GraphRewriter

`GraphRewriter` is the shared reconstruction mechanism for transforming passes.
It creates a fresh Graph, preserves every external input, copies constants only
when a retained node needs them, and records every old-to-new ValueId. A pass
can copy a node after its dependencies or replace one source Value with a newly
computed constant. `finish()` registers the mapped model outputs and validates
the reconstructed Model.

This avoids erasing entries from the Graph's index-based node/value tables. A
removed internal Value maps to `kInvalidValueId`; external inputs and outputs
must always remain mapped.

## ConstantFoldingPass

Constant Folding visits nodes in topological order and tracks values whose
Tensor contents are known. A single-output node is folded when all inputs are
known constants and `CpuBackend` supports its operator. It is evaluated through
the normal CPU execution path, then its output is inserted into the rewritten
Graph as a constant.

New folded constants are immediately available to later nodes, so a chain such
as `Add(constants) -> ReLU` folds in one pass. Nodes that depend on runtime
inputs are copied unchanged. The first implementation deliberately reuses the
CPU operator semantics instead of maintaining a second evaluator; multi-output
nodes and unsupported operators are left untouched.

## DeadCodeEliminationPass

Dead Code Elimination starts from registered model outputs and walks backwards
through Value producers. Only nodes and dependencies reached by that traversal
are copied. Unreachable nodes, unused constants, and dead intermediate Values
therefore disappear, while the complete external input interface is preserved.

The recommended initial pipeline is:

```text
ConstantFoldingPass -> DeadCodeEliminationPass
```

Folding may leave an earlier folded constant unused after a later node also
folds. DCE removes those transient constants as well as pre-existing dead
branches. Tests cover cascading folding, dead runtime-dependent branches,
ValueId-map composition, constant-only graph outputs, numerical equivalence,
and a no-change second optimization run.

## Next extension

Tensor lifetime analysis and reusable execution buffers are now implemented as
a runtime `MemoryPlan` that consumes the simplified Graph without modifying it.
See [Runtime memory planning](memory_planner.md). The next graph transformation
slice is inference-oriented operator fusion.
