# CUDA backend

CUDA is the second hardware backend after Metal. Optional `TinyInfer::cuda`
implements native resources and initial FP32 device operators on ROG.
The SDK-free `TinyInfer::cuda_contract` remains preparation-only.

Run `python tools/inspect_gpu_environment.py --output benchmark-results/phase5.0/rog-environment.json`
on ROG to record GPU/driver/toolkit availability before implementation. CPU-only
builds need no CUDA SDK. See the [Phase 5 plan](../../../docs/plans/phase5.md).

2026-10-16: optional `TinyInfer::cuda` now implements native device resources,
one non-default stream, pinned staging transfers, overlap-safe D2D and event
completion on ROG. This historical first slice did not include operators/models.
Private CUDA 12.5.1 + existing MSVC 14.29 passes real-device resource tests and
a standalone NVCC/cuBLAS precision smoke check; CPU-only builds remain isolated.
See [ROG runtime acceptance](../../../docs/acceptance/phase5.4-rog-runtime.md).

2026-10-18: `CudaDeviceRuntime::execute` accepts explicit `CudaTensorView`
inputs/output for MatMul, Gemm, FusedGemmActivation (ReLU), Add, ReLU, Tanh and
Softmax. FP32, contiguous physical layout, rank <=4 and int32 size bounds apply.
Each runtime reuses one cuBLAS handle bound to its non-default stream, with
pedantic FP32 math, host coefficient pointers and disabled atomics. User alpha/
beta are applied separately so zero coefficients do not erase NaN operands.
Device allocations must belong to the same runtime; offsets are float-aligned,
byte sizes exact, and output/input overlap is rejected. Empty output submits no
work; K=0 bypasses cuBLAS and still applies the epilogue. Completion retains
operands; submission/wait failures never replay on CPU. Single-runtime access
requires external serialization.
See [operator acceptance](../../../docs/acceptance/phase5.4-rog-operators.md).

`CudaBackend` is a capability bridge; its legacy host `execute` still throws.
`CudaExecutionPlan` / `CudaExecutionContext` in `execution_plan.h` now provide
the separate Prepared model path: immutable snapshot, shared constants uploaded
once, private context inputs/activation arena, completion-safe slot reuse,
synchronous runs and independent host outputs. Strict is default; explicit
`WholeModelCpu` selects Prepared CPU Auto for unsupported capability/unavailable
device or runtime input layout, with a reason and no per-node fallback.
CUDA allocation/upload/execution failures after runtime construction propagate,
not CPU replay. Contexts sharing a plan must be externally serialized.
See [Prepared CUDA acceptance](../../../docs/acceptance/phase5.4-rog-prepared.md).

Warm runs reuse TinyInfer device buffers and upload no constants/inputs; binding
does H2D and output retrieval does D2H. Diagnostics count TinyInfer resources,
not cuBLAS internal allocations or total process/device memory.
Softmax uses one thread per reduction group: this is a correctness baseline,
not an optimized reduction or a performance acceptance claim.

Build with `TINYINFER_ENABLE_CUDA=ON`, CUDA >=12.0, `CUDA::cudart` and
`CUDA::cublas`. ROG's verified private toolchain is selected by
`tools/build_cuda_rog.ps1`; CPU-only builds do not enable CUDA or require its SDK.
