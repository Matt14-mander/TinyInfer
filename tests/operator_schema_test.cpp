#include "tinyinfer/ops/operator_schema.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {
template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try { function(); } catch (const Exception&) { thrown = true; }
    assert(thrown);
}
}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::NodeAttributes;
    using tinyinfer::OpType;
    using tinyinfer::Shape;
    using tinyinfer::TensorSpec;
    using tinyinfer::infer_output_specs;
    using tinyinfer::operator_schema;

    assert(operator_schema(OpType::MatMul).name == "MatMul");
    assert(operator_schema(OpType::LayerNorm).minimum_inputs == 1);
    assert(operator_schema(OpType::LayerNorm).maximum_inputs == 3);

    const TensorSpec matrix{{2, 3}, DataType::Float32};
    const TensorSpec bias{{3}, DataType::Float32};
    const auto broadcast = infer_output_specs(OpType::Add, {matrix, bias});
    assert(broadcast.size() == 1);
    assert(broadcast[0] == matrix);

    const auto product = infer_output_specs(
        OpType::MatMul,
        {TensorSpec{{4, 3}, DataType::Float32},
         TensorSpec{{3, 5}, DataType::Float32}});
    assert(product[0].shape == Shape({4, 5}));

    const auto gemm = infer_output_specs(
        OpType::Gemm,
        {TensorSpec{{1, 2}, DataType::Float32},
         TensorSpec{{3, 2}, DataType::Float32},
         TensorSpec{{3}, DataType::Float32}},
        {{"transB", std::int64_t{1}}, {"alpha", 0.5F}});
    assert(gemm[0].shape == Shape({1, 3}));

    const auto unary = infer_output_specs(OpType::GELU, {matrix});
    assert(unary[0] == matrix);
    const auto softmax = infer_output_specs(
        OpType::Softmax, {matrix}, {{"axis", std::int64_t{-2}}});
    assert(softmax[0] == matrix);

    const auto layer_norm = infer_output_specs(
        OpType::LayerNorm,
        {matrix, TensorSpec{{3}, DataType::Float32},
         TensorSpec{{3}, DataType::Float32}},
        {{"epsilon", 1e-5F}});
    assert(layer_norm[0] == matrix);

    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(OpType::Add, {matrix});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(OpType::Add,
                           {matrix, TensorSpec{{2, 2}, DataType::Float32}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(
            OpType::MatMul,
            {matrix, TensorSpec{{4, 2}, DataType::Float32}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(
            OpType::Gemm,
            {TensorSpec{{1, 2}, DataType::Float32},
             TensorSpec{{3, 2}, DataType::Float32}},
            {{"transB", std::int64_t{2}}});
    });
    expect_throw<std::out_of_range>([&] {
        infer_output_specs(OpType::Softmax, {matrix},
                           {{"axis", std::int64_t{2}}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(OpType::Softmax, {matrix}, {{"axis", 1.0F}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(OpType::ReLU, {matrix},
                           {{"unknown", std::int64_t{0}}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(
            OpType::LayerNorm,
            {matrix, TensorSpec{{2}, DataType::Float32},
             TensorSpec{{3}, DataType::Float32}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(OpType::LayerNorm, {matrix},
                           {{"epsilon", -1.0F}});
    });
    expect_throw<std::invalid_argument>([&] {
        infer_output_specs(
            OpType::ReLU,
            {TensorSpec{{2, 3}, DataType::Int32}});
    });
}
