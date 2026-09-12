#include "tinyinfer/backend/cpu/matmul_kernel.h"
#include "tinyinfer/ops/matmul.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {
bool close(float a, float b) { return std::fabs(a - b) <= 1e-5F; }
void expect_equal(const tinyinfer::Tensor& a, const tinyinfer::Tensor& b) {
    assert(a.shape() == b.shape());
    for (std::size_t i = 0; i < a.numel(); ++i) assert(close(a.at(i), b.at(i)));
}
template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try { function(); } catch (const Exception&) { thrown = true; }
    assert(thrown);
}
}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::Shape;
    using tinyinfer::Tensor;
    const auto lhs = Tensor::from_vector(
        {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const auto rhs = Tensor::from_vector(
        {3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    Tensor reference({2, 2});
    Tensor strided({2, 2});
    Tensor blocked({2, 2});
    tinyinfer::cpu::matmul_reference(lhs, rhs, reference);
    tinyinfer::cpu::matmul_strided(lhs, rhs, strided);
    tinyinfer::cpu::matmul_blocked(lhs, rhs, blocked, {1, 1, 2});
    expect_equal(reference, strided);
    expect_equal(reference, blocked);
    assert(reference.at({0, 0}) == 58.0F);
    assert(reference.at({1, 1}) == 154.0F);

    auto lhs_t = Tensor::from_vector(
        {3, 2}, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
    auto rhs_t = Tensor::from_vector(
        {2, 3}, {7.0F, 9.0F, 11.0F, 8.0F, 10.0F, 12.0F});
    lhs_t.transpose(0, 1);
    rhs_t.transpose(0, 1);
    Tensor transposed_result({2, 2});
    tinyinfer::cpu::matmul_blocked(lhs_t, rhs_t, transposed_result, {8, 8, 8});
    expect_equal(reference, transposed_result);
    expect_equal(reference, tinyinfer::ops::matmul(lhs, rhs));

    const auto wide_rhs = Tensor::from_vector(
        {3, 5}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
                 11.0F, 12.0F, 13.0F, 14.0F, 15.0F});
    Tensor wide_reference({2, 5});
    Tensor wide_simd({2, 5});
    tinyinfer::cpu::matmul_reference(lhs, wide_rhs, wide_reference);
    const tinyinfer::cpu::PackedMatMulRhs packed_wide_rhs(wide_rhs);
    tinyinfer::cpu::matmul_packed_simd(lhs, packed_wide_rhs, wide_simd,
                                      {2, 5, 2});
    expect_equal(wide_reference, wide_simd);
    assert(tinyinfer::cpu::matmul_simd_width() >= 1);

    const tinyinfer::cpu::PackedMatMulRhs packed_transposed(rhs_t);
    Tensor packed_transposed_result({2, 2});
    tinyinfer::cpu::matmul_packed_simd(
        lhs_t, packed_transposed, packed_transposed_result);
    expect_equal(reference, packed_transposed_result);

    const Tensor empty_lhs({2, 0});
    const Tensor empty_rhs({0, 3});
    const auto empty_result = tinyinfer::ops::matmul(empty_lhs, empty_rhs);
    assert(empty_result.shape() == Shape({2, 3}));
    for (std::size_t i = 0; i < empty_result.numel(); ++i) assert(empty_result.at(i) == 0.0F);

    expect_throw<std::invalid_argument>([&] {
        Tensor wrong_output({2, 3});
        tinyinfer::cpu::matmul_strided(lhs, rhs, wrong_output);
    });
    expect_throw<std::invalid_argument>([&] {
        tinyinfer::cpu::matmul_blocked(lhs, rhs, blocked, {0, 1, 1});
    });
    expect_throw<std::invalid_argument>([] {
        const Tensor integers({2, 2}, DataType::Int32);
        tinyinfer::ops::matmul(integers, integers);
    });
    expect_throw<std::invalid_argument>([] {
        const Tensor vector({2});
        tinyinfer::cpu::PackedMatMulRhs packed(vector);
    });
}
