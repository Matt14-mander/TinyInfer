#include "tinyinfer/backend/cpu_backend.h"
#include <stdexcept>

namespace tinyinfer {
bool CpuBackend::supports(OpType op) const noexcept { return op == OpType::Input || op == OpType::Constant; }
void CpuBackend::execute(const Node&) { throw std::logic_error("CPU kernels are not implemented in the architecture scaffold"); }
}  // namespace tinyinfer
