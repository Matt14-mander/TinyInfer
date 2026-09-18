#pragma once

#include <filesystem>

#include "tinyinfer/model/model.h"

namespace tinyinfer {

class ModelLoader {
public:
    virtual ~ModelLoader() = default;
    virtual Model load(const std::filesystem::path& path) const = 0;
};

}  // namespace tinyinfer
