# CUDA backend

CUDA is planned as the second hardware backend after the first Metal slice on
Mac. Phase 5.0 supplies shared device/capability/completion contracts and CPU
adapters; it does not implement CUDA kernels or discover a ROG GPU remotely.

Run `python tools/inspect_gpu_environment.py --output benchmark-results/phase5.0/rog-environment.json`
on ROG to record GPU/driver/toolkit availability before implementation. CPU-only
builds need no CUDA SDK. See the [Phase 5 plan](../../../docs/plans/phase5.md).
