# ADR-0006: Own prepared CPU state and isolate invocation contexts

- Status: Accepted
- Date: 2026-09-29

## Context

Repeated graph execution sorts nodes and packs matrix RHS each invocation.
Gemm transpose preparation also deep-copies operands. A constant weight cache
needs stable identity and immutable data. A borrowed const Graph is insufficient:
a caller can mutate a constant through its Tensor Storage Buffer, and MemoryPlan
requires the identity of the graph it analyzed.

## Decision

CpuExecutionPlan owns a deep Model snapshot in shared private heap state. The
MemoryPlan is constructed after that snapshot and refers to the same stable
graph. Stored steps dispatch constant rank-2 FP32 matrix RHS using prepacked
weights; other operators use the current registry. Dynamic RHS is never cached.
Deduplicate using snapshot ValueId and effective transB, with one fixed FP32
packing format. Effective strides are interpreted directly while packing.

CpuExecutionContext keeps the state alive and privately owns an ExecutionContext.
It exposes input binding and copied outputs, without any mutable graph/constant
accessor. Contexts have independent activation storage and can be moved. Ordinary
Tensor copies retain their deep-copy semantics; prepared transA uses an explicit
private view. Gemm's existing epilogue is shared between both execution paths.

## Alternatives considered

- Borrow the original graph: cannot protect constants or graph lifetime.
- Cache by a global Tensor pointer: cannot establish snapshot/version identity.
- Add CPU packed objects to ordinary ExecutionContext: couples general runtime
  state to a backend packing format.
- Return const output references: const Tensor can still expose writable Buffer
  aliases, including when an output is a model constant.

## Consequences

Preparation incurs model copy, lifetime analysis and packing time, but warm
constant-weight runs avoid those operations. Snapshot, original constant copies,
and packed bytes increase memory usage and are documented separately. Output
extraction also copies; benchmark execution latency excludes it. New models
require new plans. Future packed formats must extend the key and eligibility
policy. In-register epilogues remain a separate measured optimization.
