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

- [Mac Phase 4.5 specialization/fusion](mac-cpu-phase4.5-fusion.md): three-mode
  paired comparison; specialization accepted locally, fusion performance pending.
  [Per-process evidence](mac-cpu-phase4.5-fusion-details.md) records all distributions.
- [ROG Phase 4.5 analysis](rog-cpu-phase4.5.md): 150 checked native/portable
  processes, component/copy diagnostics and buffer payloads.
  [Per-process evidence](rog-cpu-phase4.5-details.md) retains all timers.
- [ROG Phase 4.5 specialization/fusion](rog-cpu-phase4.5-fusion.md): 156 initial
  processes plus 18 independent control rechecks; correctness passes, native
  specialization no-bias/ReLU and fused no-op gates failed in the initial
  2026-10-09 measurements (see the targeted fix update below).
  MSVC portable is scalar width 1, not SSE2.
  [Initial/recheck evidence](rog-cpu-phase4.5-fusion-details.md) retains statistics
  and artifact hashes.
- [ROG Phase 4.5 targeted fixes](rog-cpu-phase4.5-fix.md): no-bias linear fast
  path, identity MatMul dispatch, and balanced allocation-order controls; the two
  requested native gates pass. General fusion remains experimental.
  [All intermediate and final processes](rog-cpu-phase4.5-fix-details.md) retain
  p50/p95/mean, outliers and measured source/binary hashes.
