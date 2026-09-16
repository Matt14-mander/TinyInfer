#include "tinyinfer/backend/cpu_backend.h"

namespace tinyinfer {
bool CpuBackend::supports(OpType op) const noexcept {
    return registry_.supports(op);
}

void CpuBackend::execute(const Node& node, ExecutionContext& context) {
    registry_.execute(node, context);
}
}  // namespace tinyinfer
