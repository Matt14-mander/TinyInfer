# CUDA decode validation

The worker selects serial CUDA decode by default. Passing the final worker
argument `batched`, or `NativeEngine(..., backend="cuda", batch_decode=True)`,
opts into the device validation path. Per-layer cuBLAS projections and the LM
head operate on scheduled rows. RoPE and attention use each request's position
and block table. Device scratch retains the new K/V for each layer. Logits
return in one D2H copy; each row then commits with its own COW transaction
and two pitched D2D copies. The stream synchronizes after the batch readback
and before each row's KV commit. Host staging and per-row synchronization
must be included in latency measurements.

On a supported NVIDIA GPU, use a device-required build. Substitute the
target GPU architecture for 80. Use the pinned FP32 export documented in
[the CPU measurements](acceptance/cpu-decode-batch.md).

```sh
cmake -S . -B build-device -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_MINIPREFIX=ON -DTINYINFER_ENABLE_CUDA=ON -DTINYINFER_CUDA_REQUIRE_DEVICE=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-device -j4
ctest --test-dir build-device --output-on-failure
python3 miniprefix/tools/verify_decode_batch.py build-device/miniprefix/miniprefix_cuda_worker weights/model.mpf --backend cuda
compute-sanitizer --tool memcheck --leak-check full --target-processes all --error-exitcode 1 python3 miniprefix/tools/verify_decode_batch.py build-device/miniprefix/miniprefix_cuda_worker weights/model.mpf --backend cuda
nsys profile --trace=cuda,cublas --sample=none --output=decode-pair python3 miniprefix/tools/verify_decode_batch.py build-device/miniprefix/miniprefix_cuda_worker weights/model.mpf --backend cuda
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum decode-pair.nsys-rep
```

The replay launches serial then batched workers. Inspect the steady-state
decode steps separately by process and exclude weight uploads, prefill and
warm-up when reporting decode costs. Use the kernel timeline for GEMM,
attention and other kernels, the memory timeline for H2D/D2H/D2D, and the
host API timeline for synchronization. Synchronization waits overlap device
execution, so their durations cannot be added to device times. No GPU timing
or numerical acceptance result has been collected for this implementation.

## Offline verification

On the CPU cloud host the complete Release build linked with NVCC 13.1.115,
GCC 14.2.0 and SM80. The CPU-only suite passed 56/56. In the CUDA-enabled
suite, 55 tests passed, 3 device tests skipped, and 7 failed due to the
missing CUDA driver or unavailable device. The real worker failed while
creating its first model stream. GPU numerical comparisons, memcheck and
profiling are pending a compatible NVIDIA driver and device.
