# Runtime memory planning

TinyInfer can turn a simplified Graph into a static `MemoryPlan` before
execution. The plan covers intermediate values only: model inputs remain owned
by the caller, constants remain owned by the Graph/ExecutionContext, and graph
outputs stay alive until the caller finishes reading them.

```text
optimized Graph
      |
      v
topological execution steps
      |
      v
Value lifetime intervals
      |
      v
reusable buffer slots + byte offsets
      |
      v
ExecutionContext(shared Buffer)
```

## Lifetime analysis

For each intermediate Value, `MemoryPlan` records:

- `first_step`: the step where its producer executes;
- `last_step`: the last consumer step;
- `size_bytes`: the contiguous Tensor storage requirement.

A registered graph output is extended through `node_count`, so its storage is
never reused during that invocation. Two intervals may share a slot only when
the earlier interval ends strictly before the later one starts. The strict
comparison prevents an operator output from overwriting an input while that
same operator is still reading it.

## Slot assignment

Intervals are visited in production order. The planner chooses an available
slot with the smallest required capacity growth, or creates a new slot when no
slot is free. Slot capacities are laid out with Buffer alignment inside one
shared allocation.

The plan exposes both `naive_intermediate_bytes()` and
`buffer_size_bytes()`. The first is the sum of every intermediate Tensor size;
the second is the actual shared buffer capacity, including alignment padding.
They describe memory capacity, not latency.

## Planned execution

The default `ExecutionContext(graph)` remains available and materializes normal
independent output Tensors. Planned execution is explicit:

```cpp
MemoryPlan plan(model.graph());
ExecutionContext context(model.graph(), plan);
```

CPU kernels use `prepare_output()` and write directly into their assigned
Storage range. Elementwise, MatMul, Gemm, Softmax, and LayerNorm execution no
longer need a temporary output allocation. After each node, `Executor` releases
intermediate Tensor handles whose last consumer just ran. A later Value may
then safely use the same bytes.

Consequently, dead intermediate values are not inspectable after planned
execution; only live values and registered outputs are guaranteed to remain
materialized. Repeated invocations reuse the same underlying Buffer.

## Relationship to graph optimization

Memory planning runs after Constant Folding and Dead Code Elimination:

```text
Model
  -> ConstantFoldingPass
  -> DeadCodeEliminationPass
  -> MemoryPlan
  -> planned ExecutionContext
  -> Executor
```

This ordering prevents removed nodes and constants from influencing lifetime
or capacity calculations. The planner does not modify the Graph, so it is a
runtime execution plan rather than a `GraphPass`.

## Current boundaries

- Shapes must be static, matching TinyInfer's current Graph contract.
- The planner assumes contiguous intermediate outputs.
- One CPU buffer is used; device-specific memory spaces are future Backend work.
- Slot assignment is greedy, deterministic, and safe, but not globally optimal.
- Inputs, constants, backend scratch space, and packed MatMul weights are not
  included in `naive_intermediate_bytes()`.
