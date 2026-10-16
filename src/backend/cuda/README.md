# CUDA backend

CUDA is planned as the second hardware backend after the first Metal slice on
Mac. Phase 5.0 supplies shared device/capability/completion contracts and CPU
adapters; it does not implement CUDA kernels or discover a ROG GPU remotely.

Run `python tools/inspect_gpu_environment.py --output benchmark-results/phase5.0/rog-environment.json`
on ROG to record GPU/driver/toolkit availability before implementation. CPU-only
builds need no CUDA SDK. See the [Phase 5 plan](../../../docs/plans/phase5.md).

2026-10-16: optional `TinyInfer::cuda` now implements native device resources,
one non-default stream, pinned staging transfers, overlap-safe D2D and event
completion on ROG. It is not yet an executable operator/model Backend.
Private CUDA 12.5.1 + existing MSVC 14.29 passes real-device resource tests and
a standalone NVCC/cuBLAS precision smoke check; CPU-only builds remain isolated.
See [ROG runtime acceptance](../../../docs/acceptance/phase5.4-rog-runtime.md).
