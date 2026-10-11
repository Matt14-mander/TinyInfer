#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace tinyinfer {
enum class DeviceKind { Cpu, Cuda, Metal };
struct DeviceId {
    DeviceKind kind{DeviceKind::Cpu};
    std::uint32_t ordinal{0};
};
inline bool operator==(DeviceId a, DeviceId b) noexcept {
    return a.kind == b.kind && a.ordinal == b.ordinal;
}
inline bool operator!=(DeviceId a, DeviceId b) noexcept { return !(a == b); }
std::string_view device_kind_name(DeviceKind kind);
void validate_device_id(DeviceId device);
struct DeviceInfo {
    DeviceId id;
    std::string name;
    bool available{false};
    std::string unavailable_reason;
};
} // namespace tinyinfer
