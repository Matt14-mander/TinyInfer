#include "tinyinfer/core/tensor_layout.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::Shape;
    using tinyinfer::Strides;
    using tinyinfer::TensorLayout;

    const TensorLayout contiguous({2, 3, 4});
    assert(contiguous.shape() == Shape({2, 3, 4}));
    assert(contiguous.strides() == Strides({12, 4, 1}));
    assert(contiguous.rank() == 3);
    assert(contiguous.numel() == 24);
    assert(contiguous.storage_span() == 24);
    assert(contiguous.size_bytes(DataType::Float32) == 96);
    assert(contiguous.is_contiguous());
    assert(contiguous.offset({1, 2, 3}) == 23);
    assert(contiguous.storage_offset(23) == 23);

    const auto transposed = contiguous.transposed(0, 2);
    assert(transposed.shape() == Shape({4, 3, 2}));
    assert(transposed.strides() == Strides({1, 4, 12}));
    assert(transposed.numel() == 24);
    assert(transposed.storage_span() == 24);
    assert(!transposed.is_contiguous());
    assert(transposed.offset({3, 2, 1}) == 23);
    assert(transposed.storage_offset(23) == 23);

    const TensorLayout sliced({2, 3}, {6, 2});
    assert(sliced.numel() == 6);
    assert(sliced.storage_span() == 11);
    assert(!sliced.is_contiguous());
    assert(sliced.offset({1, 2}) == 10);
    assert(sliced.storage_offset(5) == 10);

    const auto reshaped = contiguous.reshaped({2, -1});
    assert(reshaped.shape() == Shape({2, 12}));
    assert(reshaped.strides() == Strides({12, 1}));

    const TensorLayout scalar;
    assert(scalar.rank() == 0);
    assert(scalar.numel() == 1);
    assert(scalar.storage_span() == 1);
    assert(scalar.offset({}) == 0);

    const TensorLayout empty({2, 0, 4});
    assert(empty.numel() == 0);
    assert(empty.storage_span() == 0);

    expect_throw<std::invalid_argument>([] { TensorLayout({2, -1}); });
    expect_throw<std::invalid_argument>([] { TensorLayout({2, 3}, {3}); });
    expect_throw<std::invalid_argument>([] { TensorLayout({2, 3}, {3, -1}); });
    expect_throw<std::invalid_argument>([&] { contiguous.offset({1, 2}); });
    expect_throw<std::out_of_range>([&] { contiguous.offset({2, 0, 0}); });
    expect_throw<std::out_of_range>([&] { contiguous.storage_offset(24); });
    expect_throw<std::logic_error>([&] { transposed.reshaped({6, 4}); });

    const auto maximum = std::numeric_limits<std::int64_t>::max();
    expect_throw<std::overflow_error>([=] { TensorLayout({maximum, 3}); });
    expect_throw<std::overflow_error>([=] {
        TensorLayout({maximum}).size_bytes(DataType::Float32);
    });
    expect_throw<std::overflow_error>([=] {
        TensorLayout({maximum}, {3});
    });
}
