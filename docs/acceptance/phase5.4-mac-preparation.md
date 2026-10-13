# Phase 5.4 — Mac portable preparation delivery

Date: 2026-10-13. Scope authorized by user: complete Mac-development/verification
work first; native CUDA implementation and real-device validation later on ROG.

Delivered:

- SDK-free `TinyInfer::cuda_contract`, without changing the executable backend API.
- Static FP32/rank/layout/schema/int32 contract validation.
- Pointer-free row-major to column-major Gemm mapping, N/T flags, leading dimensions,
  zero-output/zero-K paths and separate user coefficient/bias/ReLU epilogue metadata.
- Independent CPU column-major interpreter vs existing CPU Graph execution:
  **773 cases**, all transpose combinations, zero/tail K, coefficient/sign-zero,
  absent/scalar/singleton/vector/row/column/full bias, empty outputs, alpha=0 with
  NaN operands and beta=0 with NaN bias.
- Invalid schema/layout/shape/dtype/rank/int32 overflow rejection and both ONNX
  fixtures' static contract coverage. Fixture inspection example.
- Extended read-only ROG inventory and a concrete implementation/handoff plan.

This is not CUDA execution: there is no native CUDA runtime, device allocation,
cuBLAS call, kernel, Prepared CUDA model or CUDA performance claim in this slice.
The CPU interpreter validates layout metadata against the documented column-major
formula; it cannot validate actual cuBLAS reduction, precision or asynchronous errors.
The 773 cases must be rerun against native GPU results on ROG.

Reproduction (no CUDA SDK/GPU required):

```bash
cmake -S . -B build-contract -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=ON
cmake --build build-contract -j 4
ctest --test-dir build-contract --output-on-failure
./build-contract/examples/tinyinfer_cuda_contract_inspect tests/fixtures/phase3_mlp_gemm.onnx
```

Current Mac SDK libc++ workaround follows the Phase 5.1 record. Final gates: CPU-only Release **44/44**, native CPU contract **1/1**,
host ASan/UBSan contract **1/1**; zero CUDA device tests are counted. Commands
and limitations are retained in `phase5.4-mac-verification.json` and logs.
Phase 5.4 remains in development; ROG work is intentionally deferred by the user.
See [the handoff plan](../plans/phase5.4.md). Phase 5.3's failed performance gate
is retained independently.
