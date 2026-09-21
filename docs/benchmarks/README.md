# Benchmark reports

This directory contains reviewed, versioned benchmark baselines. Raw JSON and
text output stays under the ignored `benchmark-results/` directory; reports
here record the environment, method, selected statistics, interpretation, and
limitations needed to reproduce a result.

## Baselines

- [Mac CPU v0.3 baseline](mac-cpu-v0.3.md): Model Runtime and MatMul results on
  an Intel Core i5-8257U MacBook Pro.

Reports from different machines are comparable only when they use the same Git
commit, Release configuration, native-architecture policy, fixture, and
benchmark arguments.
