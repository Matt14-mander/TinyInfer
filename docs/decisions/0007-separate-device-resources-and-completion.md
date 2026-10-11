# ADR-0007: Separate device resources and completion from host Tensor storage

- Status: Accepted
- Date: 2026-10-11

## Context

Tensor/Storage expose CPU-readable pointers. ExecutionContext releases handles
by host execution step, which cannot establish asynchronous device completion.
Backend OpType support alone cannot express dtype/shape/layout/device constraints.
Phase 5 needs these contracts before introducing GPU kernels.

## Decision

Keep host Tensor/Buffer semantics and existing CPU execution entry points.
Introduce DeviceId, pointer-free CapabilityQuery descriptors, and a separate
DeviceBuffer/DeviceSpan allocation abstraction. GPU resources never expose a
host pointer through this API, even when a platform supports shared storage.
Native GPU handles remain inside backend-specific subclasses. Device ordinals
identify devices within one process, not persistent hardware identity.

DeviceRuntime represents one ordered queue and owned-buffer upload/download/copy
operations. Completion is movable, cannot be copied, and retains resources until
terminal completion. Its destructor and move assignment drain outstanding work.
Explicit wait() observes sticky execution errors; destructor cleanup cannot report
errors and must not be the successful-output boundary. A backend wait() may throw
only when the operation is terminal and will never access operands again. Future
completion states must also retain native queue/device/context handles.

Allocations and queue objects are noncopyable to avoid duplicating native-handle
ownership. CpuDeviceRuntime provides synchronous allocation and byte-range copies today.
Successful request counters record cumulative payload bytes, including zero-byte
requests, rather than total resident memory or hardware allocation instrumentation.
A queue/context requires external serialization; counters are not atomic.

Add conservative Backend node-capability queries while keeping legacy
supports(OpType)/execute(Node, host ExecutionContext) source-compatible. The
default query rejects unsupported capability implementations. CpuBackend validates
schemas, attributes and provided layouts. Future GPU prepared plans own separate
device contexts rather than exposing device allocations through host Tensor.

select_model_backend() is preparation metadata: choose strict rejection or
whole-model CPU fallback, with reasons. It validates the complete CPU fallback
model. It does not execute, partition a graph, construct a GPU plan, or catch and
retry execution failures. Runtime-layout conditions remain visible and must be
enforced by a future device plan on every binding/run.

## Alternatives considered

- Place GPU pointers in Buffer/Storage: host accessors could dereference them.
- Release resources after host node submission: queued work can still reference them.
- Assume OpType support covers all shapes/dtypes: creates unreviewed capability claims.
- Per-node CPU fallback initially: introduces cross-device transfer/synchronization
  policy before the first device model path has been established.

## Consequences

No CUDA/Metal dependency is needed for CPU-only builds. Host deep-copy ownership,
CPU arithmetic and CPU Auto selection are unchanged. New APIs need recompilation;
no stable binary ABI is claimed. GPU allocation/queue/kernel adapters, runtime
layout enforcement, planned device activation reuse and weight upload lifetime
are implemented and tested in subsequent slices, not supplied by this ADR alone.
