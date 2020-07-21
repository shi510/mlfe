#pragma once
#include "mlfe/utils/onnx/proto/onnx.proto3.pb.h"
#include "mlfe/core/graph.h"
#include "mlfe/core/tensor.h"
#include <string>
#include <cstdint>
#include <memory>

namespace mlfe {
namespace onnx {

bool read_onnx_model(std::string file_name, ::onnx::ModelProto* model);

::onnx::TypeProto* make_type_proto(std::vector<int32_t> dims, int32_t type);

void fill_node_proto(node nd, ::onnx::NodeProto* nd_proto);

template <typename T, ::onnx::AttributeProto_AttributeType AttrType>
std::unique_ptr<T> get_node_proto_attr(
    const ::onnx::NodeProto* nd_proto,
    std::string name
    );

type::TypeInfo get_type_byte_size(int32_t type);

} // end namesapce onnx
} // end namespace mlfe