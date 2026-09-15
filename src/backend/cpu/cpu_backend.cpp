#include "tinyinfer/backend/cpu_backend.h"
#include <stdexcept>

namespace tinyinfer {
bool CpuBackend::supports(OpType) const noexcept { return false; }
void CpuBackend::execute(const Node&) { throw std::logic_error("CPU kernels are not implemented in the architecture scaffold"); }
}  // namespace tinyinfer
