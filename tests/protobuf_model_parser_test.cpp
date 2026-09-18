#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

void append_varint(std::string& output, std::uint64_t value) {
    while (value >= 0x80U) {
        output.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7U;
    }
    output.push_back(static_cast<char>(value));
}

void append_key(std::string& output, std::uint32_t field, std::uint8_t wire) {
    append_varint(output, (static_cast<std::uint64_t>(field) << 3U) | wire);
}

void append_varint_field(std::string& output, std::uint32_t field,
                         std::uint64_t value) {
    append_key(output, field, 0);
    append_varint(output, value);
}

void append_bytes_field(std::string& output, std::uint32_t field,
                        std::string_view value) {
    append_key(output, field, 2);
    append_varint(output, value.size());
    output.append(value.data(), value.size());
}

void append_fixed32(std::string& output, std::uint32_t bits) {
    for (unsigned index = 0; index < 4; ++index) {
        output.push_back(static_cast<char>((bits >> (index * 8U)) & 0xFFU));
    }
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string tensor_type(std::int32_t dtype,
                        const std::vector<std::int64_t>& shape) {
    std::string shape_message;
    for (const auto dimension : shape) {
        std::string dim;
        append_varint_field(dim, 1, static_cast<std::uint64_t>(dimension));
        append_bytes_field(shape_message, 1, dim);
    }
    std::string tensor;
    append_varint_field(tensor, 1, static_cast<std::uint64_t>(dtype));
    append_bytes_field(tensor, 2, shape_message);
    std::string type;
    append_bytes_field(type, 1, tensor);
    return type;
}

std::string value_info(std::string_view name, std::int32_t dtype,
                       const std::vector<std::int64_t>& shape) {
    std::string value;
    append_bytes_field(value, 1, name);
    append_bytes_field(value, 2, tensor_type(dtype, shape));
    return value;
}

std::string float_tensor(std::string_view name,
                         const std::vector<std::int64_t>& shape,
                         const std::vector<float>& values, bool raw) {
    std::string tensor;
    std::string packed_dims;
    for (const auto dimension : shape) {
        append_varint(packed_dims, static_cast<std::uint64_t>(dimension));
    }
    append_bytes_field(tensor, 1, packed_dims);
    append_varint_field(tensor, 2, 1);
    append_bytes_field(tensor, 8, name);
    std::string data;
    for (const auto value : values) append_fixed32(data, float_bits(value));
    append_bytes_field(tensor, raw ? 9U : 4U, data);
    return tensor;
}

std::string int32_tensor(std::int32_t dtype,
                         const std::vector<std::int32_t>& values) {
    std::string tensor;
    append_varint_field(tensor, 1, values.size());
    append_varint_field(tensor, 2, static_cast<std::uint64_t>(dtype));
    std::string packed;
    for (const auto value : values) {
        append_varint(packed, static_cast<std::uint64_t>(
                                  static_cast<std::int64_t>(value)));
    }
    append_bytes_field(tensor, 5, packed);
    return tensor;
}

std::string make_add_model() {
    std::string node;
    append_bytes_field(node, 1, "x");
    append_bytes_field(node, 1, "bias");
    append_bytes_field(node, 2, "y");
    append_bytes_field(node, 3, "add");
    append_bytes_field(node, 4, "Add");

    std::string graph;
    append_bytes_field(graph, 1, node);
    append_bytes_field(graph, 5,
                       float_tensor("bias", {2}, {0.5F, -0.5F}, true));
    append_bytes_field(graph, 11, value_info("x", 1, {1, 2}));
    append_bytes_field(graph, 12, value_info("y", 1, {1, 2}));

    std::string opset;
    append_varint_field(opset, 2, 17);
    std::string model;
    append_varint_field(model, 1, 8);
    append_bytes_field(model, 7, graph);
    append_bytes_field(model, 8, opset);
    return model;
}

}  // namespace

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::Tensor;
    using tinyinfer::onnx::OnnxImporter;
    using tinyinfer::onnx::ProtobufModelParser;

    ProtobufModelParser parser;
    const auto typed_float = parser.decode_tensor_proto(
        float_tensor("typed", {2}, {1.25F, -2.5F}, false));
    assert(typed_float.dtype() == DataType::Float32);
    assert(typed_float.at(0) == 1.25F);
    assert(typed_float.at(1) == -2.5F);

    const auto int32 = parser.decode_tensor_proto(
        int32_tensor(6, {7, -2}));
    assert(int32.at<std::int32_t>(0) == 7);
    assert(int32.at<std::int32_t>(1) == -2);
    const auto int8 = parser.decode_tensor_proto(
        int32_tensor(3, {-1, 127}));
    assert(int8.at<std::int8_t>(0) == -1);
    assert(int8.at<std::int8_t>(1) == 127);
    const auto float16 = parser.decode_tensor_proto(
        int32_tensor(10, {0x3C00, 0xC000}));
    assert(float16.at<std::uint16_t>(0) == 0x3C00);
    assert(float16.at<std::uint16_t>(1) == 0xC000);

    auto malformed = float_tensor("bad", {2}, {1.0F}, true);
    expect_throw<std::invalid_argument>([&] {
        parser.decode_tensor_proto(malformed);
    });

    const auto path = std::filesystem::temp_directory_path() /
                      "tinyinfer_protobuf_model_parser_test.onnx";
    {
        std::ofstream stream(path, std::ios::binary);
        const auto bytes = make_add_model();
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        assert(stream.good());
    }

    OnnxImporter importer;
    const auto model = importer.load(path);
    std::filesystem::remove(path);
    assert(model.inputs().size() == 1);
    assert(model.graph().size() == 1);

    ExecutionContext context(model.graph());
    context.bind_input(model.input_id("x"),
                       Tensor::from_vector({1, 2}, {1.0F, 2.0F}));
    CpuBackend backend;
    Executor executor(backend);
    executor.run(model.graph(), context);
    const auto& output = context.output(model.output_id("y"));
    assert(std::fabs(output.at(0) - 1.5F) < 1e-6F);
    assert(std::fabs(output.at(1) - 1.5F) < 1e-6F);

    expect_throw<std::runtime_error>([&] {
        importer.load("does-not-exist.onnx");
    });
}
