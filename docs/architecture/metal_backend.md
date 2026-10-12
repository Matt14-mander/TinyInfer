# Metal device operators — Phase 5.1

The optional `TinyInfer::metal` target implements device allocation/copies,
completion and individual FP32 operators. It consumes the Phase 5.0 contracts;
CPU-only builds remain independent of Objective-C++ and Apple frameworks.
Prepared GPU model/context APIs remain Phase 5.2 work.

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
