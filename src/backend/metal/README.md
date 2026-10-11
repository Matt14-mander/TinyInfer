# Metal backend

Phase 5.0 provides shared device/capability/completion contracts and CPU adapters.
Mac inventory on 2026-10-11 verifies Intel Iris Plus Graphics 645, command queue
creation and runtime shader library compilation. Metal is the first Phase 5.1
backend; GPU kernels and model execution are not implemented yet.

See [device contracts](../../../docs/architecture/device_runtime.md) and
[Phase 5.0 acceptance](../../../docs/acceptance/phase5.0.md). Optional Objective-C++
implementation can create shader libraries/pipelines during preparation; host
Tensor pointers must not be used to expose native Metal resource handles.
