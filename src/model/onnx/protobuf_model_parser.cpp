#include "tinyinfer/model/onnx/protobuf_model_parser.h"

#include "tinyinfer/model/onnx/import_diagnostic.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tinyinfer::onnx {
namespace {

class WireReader {
public:
    explicit WireReader(std::string_view bytes)
        : current_(reinterpret_cast<const unsigned char*>(bytes.data())),
          end_(current_ + bytes.size()) {}

    bool empty() const noexcept { return current_ == end_; }

    std::pair<std::uint32_t, std::uint8_t> key() {
        const auto value = varint();
        const auto field = static_cast<std::uint32_t>(value >> 3U);
        const auto wire = static_cast<std::uint8_t>(value & 7U);
        if (field == 0) throw std::invalid_argument("protobuf field number is zero");
        return {field, wire};
    }

    std::uint64_t varint() {
        std::uint64_t result = 0;
        for (unsigned shift = 0; shift < 64; shift += 7) {
            require(1);
            const auto byte = *current_++;
            result |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) return result;
        }
        throw std::invalid_argument("protobuf varint is too long");
    }

    std::uint32_t fixed32() {
        require(4);
        const auto value = static_cast<std::uint32_t>(current_[0]) |
                           (static_cast<std::uint32_t>(current_[1]) << 8U) |
                           (static_cast<std::uint32_t>(current_[2]) << 16U) |
                           (static_cast<std::uint32_t>(current_[3]) << 24U);
        current_ += 4;
        return value;
    }

    std::uint64_t fixed64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(current_[index]) << (index * 8U);
        }
        current_ += 8;
        return value;
    }

    std::string_view bytes() {
        const auto length = varint();
        if (length > static_cast<std::uint64_t>(end_ - current_)) {
            throw std::invalid_argument("protobuf length exceeds remaining data");
        }
        const auto size = static_cast<std::size_t>(length);
        const auto* begin = reinterpret_cast<const char*>(current_);
        current_ += size;
        return {begin, size};
    }

    void skip(std::uint8_t wire) {
        switch (wire) {
            case 0: static_cast<void>(varint()); return;
            case 1: static_cast<void>(fixed64()); return;
            case 2: static_cast<void>(bytes()); return;
            case 5: static_cast<void>(fixed32()); return;
            default:
                throw std::invalid_argument("unsupported protobuf wire type");
        }
    }

private:
    void require(std::size_t size) const {
        if (static_cast<std::size_t>(end_ - current_) < size) {
            throw std::invalid_argument("truncated protobuf message");
        }
    }

    const unsigned char* current_;
    const unsigned char* end_;
};

void require_wire(std::uint8_t actual, std::uint8_t expected,
                  const char* field) {
    if (actual != expected) {
        throw std::invalid_argument(std::string("unexpected wire type for ") + field);
    }
}

float float_from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int64_t signed_varint(std::uint64_t value) {
    return static_cast<std::int64_t>(value);
}

DataType data_type(std::int32_t value) {
    switch (value) {
        case 1: return DataType::Float32;
        case 3: return DataType::Int8;
        case 6: return DataType::Int32;
        case 10: return DataType::Float16;
        default:
            throw std::invalid_argument("unsupported ONNX TensorProto data type");
    }
}

struct ParsedTensor {
    Shape shape;
    std::int32_t data_type{0};
    std::string name;
    std::string raw_data;
    std::vector<float> float_data;
    std::vector<std::int32_t> int32_data;
    bool external{false};
};

void read_packed_varints(std::string_view bytes,
                         std::vector<std::int64_t>& output) {
    WireReader packed(bytes);
    while (!packed.empty()) output.push_back(signed_varint(packed.varint()));
}

void read_packed_int32(std::string_view bytes,
                       std::vector<std::int32_t>& output) {
    WireReader packed(bytes);
    while (!packed.empty()) {
        output.push_back(static_cast<std::int32_t>(packed.varint()));
    }
}

void read_packed_floats(std::string_view bytes, std::vector<float>& output) {
    WireReader packed(bytes);
    while (!packed.empty()) output.push_back(float_from_bits(packed.fixed32()));
}

ParsedTensor parse_tensor(std::string_view bytes) {
    ParsedTensor tensor;
    WireReader reader(bytes);
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        switch (field) {
            case 1:
                if (wire == 0) {
                    tensor.shape.push_back(signed_varint(reader.varint()));
                } else if (wire == 2) {
                    read_packed_varints(reader.bytes(), tensor.shape);
                } else {
                    require_wire(wire, 0, "TensorProto.dims");
                }
                break;
            case 2:
                require_wire(wire, 0, "TensorProto.data_type");
                tensor.data_type = static_cast<std::int32_t>(reader.varint());
                break;
            case 4:
                if (wire == 5) {
                    tensor.float_data.push_back(float_from_bits(reader.fixed32()));
                } else if (wire == 2) {
                    read_packed_floats(reader.bytes(), tensor.float_data);
                } else {
                    require_wire(wire, 5, "TensorProto.float_data");
                }
                break;
            case 5:
                if (wire == 0) {
                    tensor.int32_data.push_back(
                        static_cast<std::int32_t>(reader.varint()));
                } else if (wire == 2) {
                    read_packed_int32(reader.bytes(), tensor.int32_data);
                } else {
                    require_wire(wire, 0, "TensorProto.int32_data");
                }
                break;
            case 8:
                require_wire(wire, 2, "TensorProto.name");
                tensor.name = std::string(reader.bytes());
                break;
            case 9:
                require_wire(wire, 2, "TensorProto.raw_data");
                tensor.raw_data = std::string(reader.bytes());
                break;
            case 13:
                require_wire(wire, 2, "TensorProto.external_data");
                static_cast<void>(reader.bytes());
                tensor.external = true;
                break;
            case 14:
                require_wire(wire, 0, "TensorProto.data_location");
                tensor.external = reader.varint() != 0;
                break;
            default: reader.skip(wire); break;
        }
    }
    return tensor;
}

Tensor decode_tensor(const ParsedTensor& source) {
    if (source.external) {
        throw std::invalid_argument("external ONNX tensor data is not supported");
    }
    const auto dtype = data_type(source.data_type);
    Tensor tensor(source.shape, dtype);
    const auto count = tensor.numel();

    if (!source.raw_data.empty()) {
        if (source.raw_data.size() != tensor.size_bytes()) {
            throw std::invalid_argument("TensorProto raw_data size is inconsistent");
        }
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            source.raw_data.data());
        switch (dtype) {
            case DataType::Float32:
                for (std::size_t i = 0; i < count; ++i) {
                    const auto bits = static_cast<std::uint32_t>(bytes[i * 4]) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8U) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16U) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24U);
                    tensor.data<float>()[i] = float_from_bits(bits);
                }
                break;
            case DataType::Float16:
                for (std::size_t i = 0; i < count; ++i) {
                    tensor.data<std::uint16_t>()[i] =
                        static_cast<std::uint16_t>(bytes[i * 2]) |
                        static_cast<std::uint16_t>(bytes[i * 2 + 1] << 8U);
                }
                break;
            case DataType::Int8:
                std::memcpy(tensor.data<std::int8_t>(), bytes, count);
                break;
            case DataType::Int32:
                for (std::size_t i = 0; i < count; ++i) {
                    const auto bits = static_cast<std::uint32_t>(bytes[i * 4]) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8U) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16U) |
                        (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24U);
                    tensor.data<std::int32_t>()[i] = static_cast<std::int32_t>(bits);
                }
                break;
        }
        return tensor;
    }

    if (dtype == DataType::Float32) {
        if (source.float_data.size() != count) {
            throw std::invalid_argument("TensorProto float_data size is inconsistent");
        }
        std::copy(source.float_data.begin(), source.float_data.end(),
                  tensor.data<float>());
        return tensor;
    }
    if (source.int32_data.size() != count) {
        throw std::invalid_argument("TensorProto int32_data size is inconsistent");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = source.int32_data[i];
        switch (dtype) {
            case DataType::Float16:
                if (value < 0 || value > std::numeric_limits<std::uint16_t>::max()) {
                    throw std::invalid_argument("Float16 TensorProto value is out of range");
                }
                tensor.data<std::uint16_t>()[i] = static_cast<std::uint16_t>(value);
                break;
            case DataType::Int8:
                if (value < std::numeric_limits<std::int8_t>::min() ||
                    value > std::numeric_limits<std::int8_t>::max()) {
                    throw std::invalid_argument("Int8 TensorProto value is out of range");
                }
                tensor.data<std::int8_t>()[i] = static_cast<std::int8_t>(value);
                break;
            case DataType::Int32: tensor.data<std::int32_t>()[i] = value; break;
            case DataType::Float32: break;
        }
    }
    return tensor;
}

Shape parse_shape(std::string_view bytes) {
    Shape shape;
    WireReader reader(bytes);
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        if (field != 1) {
            reader.skip(wire);
            continue;
        }
        require_wire(wire, 2, "TensorShapeProto.dim");
        WireReader dimension(reader.bytes());
        std::optional<std::int64_t> value;
        bool symbolic = false;
        while (!dimension.empty()) {
            const auto [dim_field, dim_wire] = dimension.key();
            if (dim_field == 1) {
                require_wire(dim_wire, 0, "Dimension.dim_value");
                value = signed_varint(dimension.varint());
            } else if (dim_field == 2) {
                require_wire(dim_wire, 2, "Dimension.dim_param");
                static_cast<void>(dimension.bytes());
                symbolic = true;
            } else {
                dimension.skip(dim_wire);
            }
        }
        if (symbolic || !value) {
            throw std::invalid_argument("dynamic ONNX dimensions are not supported");
        }
        shape.push_back(*value);
    }
    return shape;
}

TensorSpec parse_type(std::string_view bytes) {
    WireReader reader(bytes);
    std::optional<TensorSpec> result;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        if (field != 1) {
            reader.skip(wire);
            continue;
        }
        require_wire(wire, 2, "TypeProto.tensor_type");
        WireReader tensor_type(reader.bytes());
        std::optional<DataType> dtype;
        std::optional<Shape> shape;
        while (!tensor_type.empty()) {
            const auto [type_field, type_wire] = tensor_type.key();
            if (type_field == 1) {
                require_wire(type_wire, 0, "TypeProto.Tensor.elem_type");
                dtype = data_type(static_cast<std::int32_t>(tensor_type.varint()));
            } else if (type_field == 2) {
                require_wire(type_wire, 2, "TypeProto.Tensor.shape");
                shape = parse_shape(tensor_type.bytes());
            } else {
                tensor_type.skip(type_wire);
            }
        }
        if (!dtype || !shape) {
            throw std::invalid_argument("incomplete ONNX tensor type");
        }
        result = TensorSpec{std::move(*shape), *dtype};
    }
    if (!result) throw std::invalid_argument("ONNX value is not a tensor type");
    return *result;
}

ValueInfo parse_value_info(std::string_view bytes) {
    WireReader reader(bytes);
    ValueInfo value;
    std::optional<TensorSpec> spec;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        if (field == 1) {
            require_wire(wire, 2, "ValueInfoProto.name");
            value.name = std::string(reader.bytes());
        } else if (field == 2) {
            require_wire(wire, 2, "ValueInfoProto.type");
            spec = parse_type(reader.bytes());
        } else {
            reader.skip(wire);
        }
    }
    if (value.name.empty() || !spec) {
        throw std::invalid_argument("incomplete ONNX ValueInfoProto");
    }
    value.spec = std::move(*spec);
    return value;
}

std::pair<std::string, AttributeValue> parse_attribute(std::string_view bytes) {
    WireReader reader(bytes);
    std::string name;
    std::optional<std::int64_t> integer;
    std::optional<float> floating;
    std::optional<std::string> string;
    std::vector<std::int64_t> integers;
    std::int32_t type = 0;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        switch (field) {
            case 1:
                require_wire(wire, 2, "AttributeProto.name");
                name = std::string(reader.bytes());
                break;
            case 2:
                require_wire(wire, 5, "AttributeProto.f");
                floating = float_from_bits(reader.fixed32());
                break;
            case 3:
                require_wire(wire, 0, "AttributeProto.i");
                integer = signed_varint(reader.varint());
                break;
            case 4:
                require_wire(wire, 2, "AttributeProto.s");
                string = std::string(reader.bytes());
                break;
            case 8:
                if (wire == 0) {
                    integers.push_back(signed_varint(reader.varint()));
                } else if (wire == 2) {
                    read_packed_varints(reader.bytes(), integers);
                } else {
                    require_wire(wire, 0, "AttributeProto.ints");
                }
                break;
            case 20:
                require_wire(wire, 0, "AttributeProto.type");
                type = static_cast<std::int32_t>(reader.varint());
                break;
            default: reader.skip(wire); break;
        }
    }
    if (name.empty()) throw std::invalid_argument("ONNX attribute has no name");
    if (type == 1 || (type == 0 && floating)) return {name, *floating};
    if (type == 2 || (type == 0 && integer)) return {name, *integer};
    if (type == 3 || (type == 0 && string)) return {name, *string};
    if (type == 7 || (type == 0 && !integers.empty())) return {name, integers};
    throw std::invalid_argument("unsupported ONNX attribute type");
}

NodeProto parse_node(std::string_view bytes) {
    WireReader reader(bytes);
    NodeProto node;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        switch (field) {
            case 1:
                require_wire(wire, 2, "NodeProto.input");
                node.inputs.emplace_back(reader.bytes());
                break;
            case 2:
                require_wire(wire, 2, "NodeProto.output");
                node.outputs.emplace_back(reader.bytes());
                break;
            case 3:
                require_wire(wire, 2, "NodeProto.name");
                node.name = std::string(reader.bytes());
                break;
            case 4:
                require_wire(wire, 2, "NodeProto.op_type");
                node.op_type = std::string(reader.bytes());
                break;
            case 5: {
                require_wire(wire, 2, "NodeProto.attribute");
                auto attribute = parse_attribute(reader.bytes());
                if (!node.attributes.emplace(std::move(attribute)).second) {
                    throw std::invalid_argument("duplicate ONNX node attribute");
                }
                break;
            }
            case 7:
                require_wire(wire, 2, "NodeProto.domain");
                node.domain = std::string(reader.bytes());
                break;
            default: reader.skip(wire); break;
        }
    }
    if (node.op_type.empty()) throw std::invalid_argument("ONNX node has no op_type");
    return node;
}

GraphProto parse_graph(std::string_view bytes) {
    WireReader reader(bytes);
    GraphProto graph;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        switch (field) {
            case 1:
                require_wire(wire, 2, "GraphProto.node");
                graph.nodes.push_back(parse_node(reader.bytes()));
                break;
            case 2:
                require_wire(wire, 2, "GraphProto.name");
                graph.name = std::string(reader.bytes());
                break;
            case 5: {
                require_wire(wire, 2, "GraphProto.initializer");
                const auto parsed = parse_tensor(reader.bytes());
                if (parsed.name.empty()) {
                    throw std::invalid_argument("ONNX initializer has no name");
                }
                try {
                    graph.initializers.push_back(
                        Initializer{parsed.name, decode_tensor(parsed)});
                } catch (const std::exception& error) {
                    OnnxImportDiagnostic diagnostic;
                    diagnostic.stage = OnnxImportStage::TensorDecode;
                    diagnostic.graph_name = graph.name;
                    diagnostic.value_name = parsed.name;
                    diagnostic.message = error.what();
                    throw OnnxImportError(std::move(diagnostic));
                }
                break;
            }
            case 11:
                require_wire(wire, 2, "GraphProto.input");
                graph.inputs.push_back(parse_value_info(reader.bytes()));
                break;
            case 12:
                require_wire(wire, 2, "GraphProto.output");
                graph.outputs.push_back(parse_value_info(reader.bytes()));
                break;
            default: reader.skip(wire); break;
        }
    }
    return graph;
}

std::pair<std::string, std::int64_t> parse_opset(std::string_view bytes) {
    WireReader reader(bytes);
    std::string domain;
    std::int64_t version = 0;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        if (field == 1) {
            require_wire(wire, 2, "OperatorSetIdProto.domain");
            domain = std::string(reader.bytes());
        } else if (field == 2) {
            require_wire(wire, 0, "OperatorSetIdProto.version");
            version = signed_varint(reader.varint());
        } else {
            reader.skip(wire);
        }
    }
    return {domain, version};
}

ModelProto parse_model(std::string_view bytes) {
    WireReader reader(bytes);
    ModelProto model;
    bool has_graph = false;
    while (!reader.empty()) {
        const auto [field, wire] = reader.key();
        if (field == 7) {
            require_wire(wire, 2, "ModelProto.graph");
            model.graph = parse_graph(reader.bytes());
            has_graph = true;
        } else if (field == 8) {
            require_wire(wire, 2, "ModelProto.opset_import");
            const auto [domain, version] = parse_opset(reader.bytes());
            if (domain.empty() || domain == "ai.onnx") {
                model.opset_version = version;
            }
        } else {
            reader.skip(wire);
        }
    }
    if (!has_graph) throw std::invalid_argument("ONNX model has no graph");
    if (model.opset_version == 0) {
        throw std::invalid_argument("ONNX model has no default-domain opset");
    }
    return model;
}

}  // namespace

ModelProto ProtobufModelParser::parse(const std::filesystem::path& path) const {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        OnnxImportDiagnostic diagnostic;
        diagnostic.stage = OnnxImportStage::FileRead;
        diagnostic.model_path = path;
        diagnostic.message = "failed to open ONNX model";
        throw OnnxImportError(std::move(diagnostic));
    }
    const std::string bytes((std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
    if (stream.bad()) {
        OnnxImportDiagnostic diagnostic;
        diagnostic.stage = OnnxImportStage::FileRead;
        diagnostic.model_path = path;
        diagnostic.message = "failed to read ONNX model";
        throw OnnxImportError(std::move(diagnostic));
    }
    try {
        return parse_model(bytes);
    } catch (const OnnxImportError& error) {
        auto diagnostic = error.diagnostic();
        diagnostic.model_path = path;
        throw OnnxImportError(std::move(diagnostic));
    } catch (const std::exception& error) {
        OnnxImportDiagnostic diagnostic;
        diagnostic.stage = OnnxImportStage::ProtobufParse;
        diagnostic.model_path = path;
        diagnostic.message = error.what();
        throw OnnxImportError(std::move(diagnostic));
    }
}

Tensor ProtobufModelParser::decode_tensor_proto(
    std::string_view serialized) const {
    ParsedTensor parsed;
    try {
        parsed = parse_tensor(serialized);
        return decode_tensor(parsed);
    } catch (const OnnxImportError&) {
        throw;
    } catch (const std::exception& error) {
        OnnxImportDiagnostic diagnostic;
        diagnostic.stage = OnnxImportStage::TensorDecode;
        diagnostic.value_name = parsed.name;
        diagnostic.message = error.what();
        throw OnnxImportError(std::move(diagnostic));
    }
}

}  // namespace tinyinfer::onnx
