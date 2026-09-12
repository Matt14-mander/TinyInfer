# MatMul benchmark

Build benchmarks in Release mode so compiler optimization is enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_BENCHMARKS=ON \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build --target tinyinfer_matmul_benchmark
./build/benchmarks/tinyinfer_matmul_benchmark
```

The default square sizes are 64, 128, 256, and 512. Custom sizes can be
passed as command-line arguments. The allocation cost is excluded: each
kernel writes into a preallocated output Tensor. Reference results are timed
through size 128 because its indexed implementation intentionally prioritizes
clarity; larger results are checked against the stride-aware kernel.

Output reports average milliseconds and GFLOP/s for reference, stride-aware,
cache-blocked, and packed SIMD kernels. Packed SIMD timing reuses an RHS packed
before the timed loop, modeling inference where constant model weights are
packed once and reused. Every run verifies the optimized outputs before it
finishes.

`TINYINFER_ENABLE_NATIVE_ARCH` is optional and disabled by default so release
binaries remain portable. Enabling it lets x86 builds select AVX2/FMA when the
host supports them; baseline x86-64 uses SSE2, and ARM builds use NEON.
