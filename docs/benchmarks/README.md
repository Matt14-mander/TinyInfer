# Benchmark reports

This directory contains reviewed, versioned benchmark baselines. Raw JSON and
text output stays under the ignored `benchmark-results/` directory; reports
here record the environment, method, selected statistics, interpretation, and
limitations needed to reproduce a result.

## Baselines

- [Mac CPU v0.3 baseline](mac-cpu-v0.3.md): Model Runtime and MatMul results on
  an Intel Core i5-8257U MacBook Pro.
- [ROG CPU v0.3 baseline](rog-cpu-v0.3.md): correctness coverage, Model Runtime,
  and MatMul results on an Intel Core i9-13980HX ROG machine.
- [Mac CPU Phase 4.2 acceptance](mac-cpu-phase4.2.md): paired graph optimization
  measurements and an unchanged ONNX control on the Intel Mac.

- [ROG CPU Phase 4.3 acceptance](rog-cpu-phase4.3.md): Gemm + ReLU fusion.
- [Mac CPU Phase 4.4 measurements](mac-cpu-phase4.4.md): prepared execution,
  preparation costs, static weights and dynamic fallback controls.
- [ROG CPU Phase 4.4 acceptance](rog-cpu-phase4.4.md): portable/native 40/40
  correctness gates, 24 scenarios per configuration, preparation components,
  persistent payload memory and performance interpretation.
- [ROG Phase 4.4 detailed statistics](rog-cpu-phase4.4-details.md): all 144
  processes, component costs, context creation and first execution.

- [Mac CPU Phase 4.5 analysis](mac-cpu-phase4.5.md): prepared execution components,
  bias/tail controls, application copies, and implementation priorities.
  [Per-process evidence](mac-cpu-phase4.5-details.md) retains p50/p95/mean.

Reports from different machines are comparable only when they use the same Git
commit, Release configuration, native-architecture policy, fixture, and
benchmark arguments.
