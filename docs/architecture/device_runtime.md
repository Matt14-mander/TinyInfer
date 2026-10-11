# Device runtime contracts (Phase 5.0)

Phase 5.0 provides CPU implementations and contracts for future GPU backends.
It does not implement CUDA/Metal kernels or GPU model execution.

## Capability and model selection

`Backend::device_info()` identifies an available device or an unavailable reason.
`query_capability(CapabilityQuery)` inspects a copied Node (including attributes),
TensorSpecs, optional current strides, and requested DeviceId. No Tensor/storage
pointer is exposed. Unknown strides mean preparation has not observed runtime
bindings, not that every layout is contiguous. A backend must return
`requires_runtime_layout_check` if its support depends on those bindings.

CpuBackend validates OpType registration, operand/output arity, FP32 schema,
shape inference, attributes, and valid supplied input strides; supplied planned
output strides must be contiguous. CPU supports its valid strided inputs and
therefore does not require a separate layout eligibility decision. Invalid
queries return unsupported with a reason; allocator/resource failures propagate.

`select_model_backend(model, backend, policy)` validates the model and returns
ModelPlacement. Strict rejects an unavailable device/unsupported node before
execution. WholeModelCpu selects cpu:0 only after checking all CPU nodes. It
retains requested/effective device, fallback reason and available node diagnostics.
Unavailable-device diagnostics can have an empty node list. This is selection
metadata; the caller prepares the effective plan (currently CpuExecutionPlan).
Supported nodes with runtime layout conditions remain conditional. No error
from a later execute/copy/wait is interpreted as permission to rerun on CPU.

Existing Executor and CpuExecutionPlan entry points retain their host execution
behavior and do not query capabilities inside each warm node dispatch. The new
preflight is consumed explicitly; it is not a silent policy change.

## Owned resources and copies

`DeviceBuffer` carries DeviceId, byte capacity and alignment. `DeviceSpan` retains
that allocation and identifies an offset/length with overflow-safe range checks.
A GPU DeviceBuffer always returns null from host_data(); native handles can be
accessed only by its backend subclass. It cannot be inserted into host Storage.
Allocations and runtime queue objects cannot be copied; resource handles use
shared ownership. CPU allocations retain their allocator and survive runtime
destruction.

```cpp
CpuDeviceRuntime queue;
auto host = std::make_shared<Buffer>(16);
auto device = queue.allocate(16);
queue.upload({device, 0, 16}, host).wait();
auto output = std::make_shared<Buffer>(16);
queue.download(output, {device, 0, 16}).wait();
```

Uploads/downloads take owned host Buffer handles, with byte offsets. Copy spans
must have equal byte lengths and belong to the queue device. CpuDeviceRuntime
uses synchronous memmove, including overlap within an allocation; zero-byte
requests validate operands/ranges but never perform null-pointer arithmetic.
A future device copy adapter must implement the same overlap semantics or
reject overlapping ranges explicitly. It must not silently use undefined native
copy behavior. Cross-device copies are outside the first contract slice.

DeviceRuntimeStatistics counts successful allocations/uploads/downloads/copies
and cumulative payload bytes. Invalid requests do not increment counters. Zero
bytes count as a logical successful request. These are not resident-memory,
allocator-capacity, native-transfer, or completed-kernel counters. Future async
adapters must document whether submission counters include terminal failures.

## Completion and error boundary

Each device runtime is one ordered queue; concurrent access requires external
serialization. Completion retains operands (and future queue state), including
when callers drop their own handles. ready() means terminal, not successful.
Explicit wait() is required to observe failure before publishing host output.
Backend CompletionState::wait() returns/throws only once work is terminal and
cannot access operands anymore; repeated waits must preserve a terminal error.

Destructor and move assignment wait for pending work before releasing retained
resources. Cleanup is noexcept and suppresses errors; it protects lifetimes but
cannot replace explicit successful completion. The public GPU inference entry
point in later phases remains synchronous until an explicit async API is designed.
Host-step MemoryPlan release alone cannot authorize asynchronous reuse. A later
device plan must also respect queue ordering/dependencies and completion lifetime.

## Evidence and next backend

See [Phase 5.0 acceptance](../acceptance/phase5.0.md),
[ADR-0007](../decisions/0007-separate-device-resources-and-completion.md), and
[Phase 5 plan](../plans/phase5.md). Local Metal device/queue/library discovery is
an environment prerequisite, not GPU arithmetic acceptance. Device-lifetime
mock tests do not establish real asynchronous hardware correctness.
