# Graph optimizer

The Phase 4 optimizer starts with a model-aware `GraphPass` pipeline:

- `OptimizationResult` owns the output Model, ValueId mapping, and statistics.
- `PassManager` validates interfaces and composes mappings across passes.
- `NoOpPass` verifies the complete pipeline without transforming the Graph.

See `docs/architecture/graph_optimizer.md` for the invariants and planned graph
reconstruction layer. Constant Folding and Dead Code Elimination are the next
transforming passes.
