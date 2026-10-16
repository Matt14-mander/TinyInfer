# Phase 5.4 — CUDA second backend

Date: 2026-10-13 (Asia/Shanghai). Baseline: `d01126a`, Phase 5.3 transfer follow-up.
Status: Mac preparation and first ROG native resource-runtime slice implemented;
CUDA operator/model integration and performance remain open. Phase 5.3 Metal
performance gates remain unmet and unchanged.

2026-10-16 update: ROG inventory and SDK-free Release baseline are complete
(45/45 tests, 773 CPU interpretation cases). Native implementation is pending
a compatible Toolkit configuration: only CUDA 10.2 was found, incompatible with
the default MSVC 19.43 and lacking the planned pedantic math API. No installation
or native GPU acceptance is claimed. See [ROG preflight](../acceptance/phase5.4-rog-preflight.md).

Later 2026-10-16 update supersedes that preflight blocker: authorized private
CUDA 12.5.1 + existing MSVC 14.29 now builds and runs on RTX 4070 without a driver
update. Optional native resource runtime, owned transfers and event completion
pass 47/47 full tests; CPU-only passes 45/45. Both native tests pass Compute
Sanitizer memcheck/full leak checks. Operator/model integration remains pending;
the standalone NVCC/cuBLAS probe is not a production operator acceptance.
See [ROG first runtime slice](../acceptance/phase5.4-rog-runtime.md).

## Goal and sequence

Run the same static FP32 model under CPU, Metal and CUDA contracts. Phase 5.4 is
complete only after ROG native CUDA correctness, lifecycle and performance evidence.

| Slice | Work | Exit boundary |
|---|---|---|
| Mac preparation (this delivery) | SDK-free operator contract, cuBLAS mapping metadata, CPU interpretation tests, inventory and ROG handoff | Pure C++ tests pass; no CUDA availability or GPU execution claim |
| ROG runtime | Optional CUDA target, device inventory, device allocations, ordered non-default stream, pinned staging, event completion | Real H2D/D2H/D2D offsets/overlap/zero/lifetime/failure tests |
| ROG operators | cuBLAS FP32 product, bias/activation epilogue; Add/ReLU/Tanh/Softmax kernels | Real single-op CPU comparisons, special-value and tail tests |
| Prepared CUDA model | Constant upload once, private context buffers, completion-safe arena reuse, explicit whole-model CPU Auto fallback | Both fixtures, changing inputs/RHS/bias, context isolation/moves/lifetime and output ownership |
| Cross-backend/performance | Same case definitions, separate device reports, independent processes | All semantic gates pass; preparation/transfer/completed run/E2E p50/p95 and memory published |

## Implemented portable contract

`TinyInfer::cuda_contract` depends only on the existing CPU core and C++17.
`cuda::validate_operator_contract()` validates the prospective static operator
subset and returns runtime-layout requirements. It is **not** a Backend and must
not be used to select an executable CUDA placement. Success says nothing about
driver availability, device allocation limits or native kernel implementation.
No CUDA headers, GPU mocks, kernels or vendor calls are compiled by this target.

Initial scope: FP32 contiguous physical layout, rank <=4,
MatMul/Gemm/FusedGemmActivation (ReLU), Add/ReLU/Tanh/Softmax. Schemas use existing
TinyInfer inference. Each dimension and element count is conservatively bounded
by int32 for first-version kernel indices/cuBLAS parameters. Larger shapes,
noncontiguous moved inputs and other operators must reject or trigger an explicit
whole-model CPU fallback in the later executable backend. Actual device/resource
limits and runtime spans must be checked by the ROG implementation.

`GemmDescriptor` maps row-major output memory to a column-major transposed result:

```
C^T = op(B)^T * op(A)^T
native m=N, n=M, k=K
left=B, right=A; trans_left=transB, trans_right=transA
lda=max(1, original B columns)
ldb=max(1, original A columns)
ldc=max(1, N)
```

Empty output skips dispatch; zero K must synthesize a positive-zero raw product
and still run the required user epilogue. The prospective vendor product uses
alpha=1/beta=0. User alpha/beta/bias/ReLU remain in a separate epilogue so alpha=0
or beta=0 does not silently remove operand/bias NaN semantics of the CPU contract.
These policies are preparation decisions, not proof of cuBLAS special-value
behavior; GPU tests must verify them on ROG.

cuBLAS documents column-major matrix storage and leading dimensions in its
[official manual](https://docs.nvidia.com/cuda/cublas/index.html#data-layout).
The same manual describes [pedantic math mode](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t).
The ROG first implementation will explicitly request prescribed FP32 precision
and avoid default TF32/reduced-precision substitutions; verify the installed
Toolkit API before freezing native calls. No new lower-level compiler IR is
introduced for this metadata.

## ROG handoff

Run in a Visual Studio developer terminal on Windows (or a compiler-enabled
shell on Linux):

```bash
python tools/inspect_gpu_environment.py --output rog-gpu-environment.json
```

The collector records `nvidia-smi`, GPU model/driver/memory, `nvcc --version`,
CMake, compiler commands and Windows MSVC path/CPU inventory. Missing commands
are reported and do not install packages or change drivers. `nvidia-smi`'s CUDA
version is not evidence that the Toolkit/NVCC is installed. Hardware/compiler
inventory is not native TinyInfer CUDA acceptance.

First run the portable suite on ROG, before adding native sources:

```bash
cmake -S . -B build-cuda-contract -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=ON
cmake --build build-cuda-contract --config Release
ctest --test-dir build-cuda-contract -C Release --output-on-failure
```

A future executable `TinyInfer::cuda` target will be optional and linked through
[CMake FindCUDAToolkit](https://cmake.org/cmake/help/v3.29/module/FindCUDAToolkit.html)
(`CUDA::cudart`, `CUDA::cublas`) with CUDA language enabled for .cu kernels. It
is not present in this Mac delivery. Do not advertise a working native CUDA
build option before those sources and device tests exist.

## Native correctness/ownership requirements

- Runtime/device identity is validated per instance, not merely GPU ordinal.
  Public host Tensor pointers never represent device memory.
- One ordered stream; event/completion errors are sticky. Operands, pinned
  staging, stream/device resources survive pending work and wrapper destruction.
- `ready()` cannot publish a download early; explicit `wait()` must establish
  terminal completion before success or failure releases referenced resources.
  Stream/context failure handling requires testing against the actual Toolkit;
  no automatic CPU replay of failed GPU work.
- Host/public inference is synchronous. Constants/pipelines/handle preparation
  are separate from repeat runs. Pinned staging reuse requires completion leases.
- Prepared contexts share immutable constants but have independent writable
  inputs/activations. Arena slots cannot be reused based only on host lifetimes.
- Strict is default; allowed fallback is whole-model Prepared CPU Auto with a
  reason. Runtime layout fallback preserves CPU Tensor ownership/strides.
- Use unchanged fixture tolerance 1e-5 and matrix tolerance
  1e-5*(1+abs(reference)), with NaN/Inf/signed-zero, zero K/empty dimensions,
  transpose/bias/negative coefficients/alpha=0/beta=0, tails, changed bindings,
  independent output copies and asynchronous cleanup tests.

## Performance boundary

Before measuring ROG, freeze its target hardware and target cases/thresholds in a
separate record. Reuse the Metal model workload definitions and measurement
ownership: cold preparation, first inference, H2D/D2H, completed resident execution,
independent host-to-host E2E and whole-model CPU control; record p50/p95, allocations,
transfer bytes and all process data. Compare with native Prepared CPU Auto on the
same ROG, not Mac latencies. Do not infer CUDA speedups from portable tests or
use ROG results to relabel the failed Mac Metal Phase 5.3 gate.
