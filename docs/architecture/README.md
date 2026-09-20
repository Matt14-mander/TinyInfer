# Architecture documentation

This directory is the source of truth for TinyInfer's current implementation.
It describes module boundaries, data flow, supported behavior, and known
limitations. These documents should change when the code changes.

## Contents

- [System architecture](architecture.md): layers and module responsibilities.
- [Tensor fundamentals](tensor.md): layout, storage, views, iteration, and
  Tensor value semantics.
- [ONNX importer architecture](onnx_importer.md): parsing, translation, Gemm,
  and the currently supported model subset.
- [Development roadmap](roadmap.md): phase status, boundaries, and exit
  criteria.

Architecture documents explain the resulting system. Explanations of how to
rebuild a feature belong in `../tutorials`, while the reasoning behind a
non-obvious choice belongs in `../decisions`.
