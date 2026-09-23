# Graph optimizer

The Phase 4 optimizer starts with a model-aware `GraphPass` pipeline:

- `OptimizationResult` owns the output Model, ValueId mapping, and statistics.
- `PassManager` validates interfaces and composes mappings across passes.
- `NoOpPass` verifies the complete pipeline without transforming the Graph.
- `GraphRewriter` safely rebuilds index-based Graphs and maintains ValueId maps.
- `ConstantFoldingPass` evaluates supported constant-only CPU subgraphs.
- `DeadCodeEliminationPass` retains only nodes reachable from model outputs.

See `docs/architecture/graph_optimizer.md` for reconstruction invariants,
pass ordering, current limitations, and verification coverage.
