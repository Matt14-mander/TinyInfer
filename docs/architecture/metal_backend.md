# Metal device operators — Phase 5.1

The optional `TinyInfer::metal` target implements device allocation/copies,
completion and individual FP32 operators. It consumes the Phase 5.0 contracts;
CPU-only builds remain independent of Objective-C++ and Apple frameworks.
Prepared model/context APIs are implemented in Phase 5.2 (described below).
The frozen Intel Mac performance target is accepted in the
[Phase 5.3 closeout](../acceptance/phase5.3-closeout.md); it does not imply that
every supported model is faster on Metal.

## Entry point

Include `tinyinfer/backend/metal/metal_runtime.h` and link `TinyInfer::metal`.
`MetalDeviceRuntime` enumerates macOS devices, creates one ordered command queue,
compiles a runtime shader library and prepares five compute pipelines once.
No offline `metal` compiler is required. Initialization failures are explicit.
`MetalBackend` supplies node-level capability queries; its legacy host
`execute(Node, ExecutionContext)` rejects use. GPU memory is never exposed as
host Tensor storage.

`MetalTensorView` carries a TensorSpec, owned DeviceSpan and actual strides.
`execute(op, inputs, output, attributes)` validates schema, physical layout,
exact view byte length, float offset alignment, ownership domain and output
non-aliasing before encoding a command. It returns Completion; callers wait
before consuming host output. See [the example](../../examples/metal_ops.cpp).

## Accepted initial support

| Operator | Scope |
| --- | --- |
| Add | FP32, contiguous physical inputs/output, multidirectional broadcasting, rank <=4 |
| ReLU / Tanh | FP32 contiguous tensors, rank <=4 |
| MatMul | FP32 rank-2 matrices |
| Gemm / FusedGemmActivation | Rank-2 FP32, transA/transB, alpha/beta, absent/scalar/vector/row/column/full broadcast bias, None/ReLU |
| Softmax | FP32 contiguous rank 1–4, any schema-valid axis, including non-final axes |

Tensor count/dimensions must fit uint32 and byte payload must fit the device's
maxBufferLength. Unsupported operators and noncontiguous physical layouts are
rejected. Unknown preparation strides are marked as requiring runtime checks;
every dispatch checks actual views. This is not a general ONNX/GPU operator
compatibility claim. The public model importer profile is unchanged.

Each output element of the initial matrix kernel performs a serial K reduction;
it is a correctness implementation, not an optimized GEMM or performance claim.
Softmax uses one thread per reduction group, a stable max shift, then exp/sum.
Modern SDK/OS uses safe math plus precise FP32 functions; older supported SDK/OS
uses fastMathEnabled=false. FP contraction is disabled in the shader source.
ReLU's ordered-positive mask preserves CPU NaN/+0 behavior. The zero-K product
still applies the epilogue. Zero output needs no dispatch; zero-length input
spans bind a dummy resource, never a native end-of-buffer offset.

## Memory, copies and ownership

Logical device allocations use private MTLBuffer storage. Public host_data()
remains null. Logical zero-byte allocations have a small physical dummy buffer;
staging/scratch buffers are internal resources.

Uploads snapshot an owned host Buffer into a shared staging buffer, then encode
a blit to private storage. Downloads blit into shared staging and publish host
bytes only on completed work. wait(), ready() after terminal GPU status, and
synchronize() all finish required host publication. Overlapping device copies
use a scratch resource for memmove semantics, including unaligned byte ranges.
Allocations/copies/views belong to one runtime instance; another queue's
allocation is rejected even if its device ordinal matches. Cross-queue resource
sharing and dependencies are not implemented.

Completion retains C++ operands; native commands retain referenced Metal
resources. Each completion also keeps native device/queue handles alive.
Allocations retain runtime state and survive wrapper destruction. Pending
operations have no C++ shared-ownership cycle with runtime state. Destructor
cleanup drains work. Explicit wait() observes sticky errors; queue errors poison
subsequent submissions, without automatic CPU retry. All queue/completion access
requires external serialization.

## Diagnostics

- DeviceRuntimeStatistics: successful logical allocations/copy submissions and
  cumulative payload bytes. Internal staging/scratch allocations are excluded;
  counters are not resident VRAM or completed-work instrumentation. A submission
  that later fails is still counted; terminal errors must be checked separately.
- MetalKernelStatistics: five preparation pipeline creations; dispatches count
  actual nonempty compute command submissions. Zero-output operations do not
  increment dispatches. Neither counter measures GPU execution time.

## Build and actual-device tests

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_ENABLE_METAL=ON -DTINYINFER_METAL_REQUIRE_DEVICE=ON \
  -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=ON
cmake --build build-metal -j 4
MTL_DEBUG_LAYER=1 ctest --test-dir build-metal --output-on-failure
./build-metal/examples/tinyinfer_metal_ops
```

Use a normal host developer terminal with GPU access. A sandbox may enumerate
no devices. REQUIRE_DEVICE=ON makes this a failure; OFF allows unavailable-GPU
tests to return skip code 77. Skips are never GPU acceptance evidence. Enabling
Metal on a non-macOS system fails configuration; default Metal OFF needs no GPU SDK.
See [the acceptance record](../acceptance/phase5.1.md) for this workspace's SDK
include workaround, hardware, sanitizer boundary and retained failed checks.

## Prepared model execution (Phase 5.2)

`MetalExecutionPlan` snapshots a Model, prepares its order and MemoryPlan, and
uploads needed constants once. `create_context()` allocates independent input
buffers and an activation arena. Device values persist between operations.
Before slot reuse the executor waits all completions through the old value's
last consumer; every run ends with terminal completion checks. This conservative
wait policy prioritizes correctness; queue batching and reduced waits require
separate performance work.

`bind_input()` and `output()` perform synchronous owned copies; `run()` returns
only after completion. Returned outputs are independent host Tensors. Constants,
graph and device storage remain private. A context retains its plan; sharing a
plan across threads requires external serialization. Binding a Tensor lvalue
materializes a contiguous snapshot; moving a strided view retains its layout.

Strict is the default. WholeModelCpu prepares an additional CPU Auto plan and
falls back for unsupported static capability or a moved noncontiguous input.
Fallback is whole-model, with a reported placement reason; GPU execution failures
propagate. A later contiguous rebind can restore GPU execution. The contract also
applies to input-only/constant-only graph outputs. See the
[Phase 5.2 acceptance](../acceptance/phase5.2.md).

### Input ownership refinement in Phase 5.3

Performance analysis found an extra deep Tensor copy when binding the CPU backup.
The by-value binding parameter already owns the required lvalue snapshot; after
any synchronous upload it is moved into the CPU context. Metal contexts retain
bound/layout flags rather than another host input Tensor. Strict GPU contexts
can release host input storage after upload completion. This preserves lvalue
isolation and moved-view layouts, while removing the duplicate CPU fallback copy.
A counting-allocator regression test checks exactly one lvalue copy and zero
additional allocations for a moved fallback input.

### Transfer reuse and optional command profiling

Prepared uploads read the owned Tensor buffer at its storage byte offset; owned
outputs download directly into the new Tensor buffer. The intermediate C++
Buffer allocation/copy at both boundaries is removed. The native Metal shared
staging buffer remains necessary for private device resources.

Each ordered runtime caches at most two native staging buffers. A lease remains
held until command terminal completion and, for downloads, host publication.
Busy cached stages are never overwritten; additional in-flight requests use
uncached buffers. Free cached stages can grow for larger transfers. Warm fixed-size
model inference reuses staging allocations. `transfer_statistics()` reports
native staging allocations, reuse and cached capacity separately from logical
DeviceRuntimeStatistics. Cached capacity is retained until runtime destruction.

Optional profiling records completed command GPU timestamp durations (when the
device provides them) and host wait durations. Profiling is disabled by default
and excluded from acceptance runs. GPU durations include each submitted compute
or blit command, not a pure kernel-only hardware duration. Host waiting includes
completion scheduling; its difference from GPU time is not an exact queue-delay
measurement. Profiling/reset and all cache access require external serialization.
