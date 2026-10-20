# Metal backend — Phase 5.1–5.3

The optional `TinyInfer::metal` target now implements owned device allocations,
staged transfers, completion and initial FP32 device operators on macOS.
The Mac Intel Iris Plus Graphics 645 passed actual-device operator/transfer tests.
Prepared model execution is accepted in Phase 5.2. The original frozen-target
performance gates passed on 2026-10-20; see the
[closeout record](../../../docs/acceptance/phase5.3-closeout.md). Small models still favor CPU.

Include `tinyinfer/backend/metal/metal_runtime.h`, link `TinyInfer::metal`, and
use explicit MetalTensorView/Completion. The legacy host Executor is rejected.
Runtime library and five pipelines are prepared once; no offline metal tool is needed.

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DTINYINFER_ENABLE_METAL=ON -DTINYINFER_METAL_REQUIRE_DEVICE=ON
cmake --build build-metal -j 4
MTL_DEBUG_LAYER=1 ctest --test-dir build-metal --output-on-failure
./build-metal/examples/tinyinfer_metal_ops
```

Run in a host terminal with GPU visibility. Default Metal OFF builds CPU-only.
See [architecture](../../../docs/architecture/metal_backend.md),
[acceptance](../../../docs/acceptance/phase5.1.md), and the
[example](../../../examples/metal_ops.cpp) for scopes and SDK details.
