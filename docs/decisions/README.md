# Design decisions

This directory contains lightweight Architecture Decision Records (ADRs).
Each record captures the context and tradeoffs behind a decision that is not
obvious from the final code. Records are historical: when a choice changes,
the old record is marked superseded and a new record is added.

## Naming

Use a sequential number and a short description:

```text
0001-separate-storage-from-tensor.md
0002-share-storage-in-tensor-views.md
```

## Record template

```markdown
# ADR-NNNN: Decision title

- Status: Proposed | Accepted | Superseded
- Date: YYYY-MM-DD

## Context

What problem or constraint required a decision?

## Decision

What did TinyInfer choose?

## Alternatives considered

What other approaches were evaluated?

## Consequences

What becomes easier, harder, or deliberately deferred?
```

## Initial decision backlog

- Separate Tensor metadata from Buffer and Storage ownership.
- Keep Tensor copies deep while views explicitly share storage.
- Centralize elementwise traversal in TensorIterator and kernel runners.
- Represent graph edges as Values rather than connecting Nodes directly.
- Separate ONNX protobuf parsing from graph import.
- Keep Executor independent of CPU operator implementations through Backend
  and OperatorRegistry boundaries.

The backlog is not yet a set of accepted ADRs. Each item should be reconstructed
and reviewed against the implementation before receiving a numbered record.
