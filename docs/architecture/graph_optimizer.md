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

## Next extension

The next slice is a reusable Graph reconstruction helper. It will copy inputs
and constants, visit nodes in topological order, and maintain the old-to-new
ValueId map used by Constant Folding and Dead Code Elimination. Stable vector
IDs will not be edited in place.
