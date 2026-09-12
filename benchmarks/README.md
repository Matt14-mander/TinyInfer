# MatMul benchmark

Build benchmarks in Release mode so compiler optimization is enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_BENCHMARKS=ON
cmake --build build --target tinyinfer_matmul_benchmark
./build/benchmarks/tinyinfer_matmul_benchmark
```

The default square sizes are 64, 128, 256, and 512. Custom sizes can be
passed as command-line arguments. The allocation cost is excluded: each
kernel writes into a preallocated output Tensor. Reference results are timed
through size 128 because its indexed implementation intentionally prioritizes
clarity; larger results are checked against the stride-aware kernel.

Output reports average milliseconds and GFLOP/s for reference, stride-aware,
and cache-blocked kernels. Every run verifies the optimized outputs before it
finishes.
