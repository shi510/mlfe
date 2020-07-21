#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/utils/onnx/onnx_registry.h"
#include <fstream>
#include <string>
#include <vector>

namespace mlfe {
namespace onnx {
using namespace ::onnx;

bool read_onnx_model(std::string file_name, ::onnx::ModelProto* model)
{
    std::ifstream file(file_name, std::ios_base::binary);
    auto status = model->ParseFromIstream(&file);
    file.close();
    return status;
}

::onnx::TypeProto* make_type_proto(std::vector<int32_t> dims, int32_t type)
{
    auto tp = new ::onnx::TypeProto;
    auto tt = new ::onnx::TypeProto_Tensor;
    auto ts = new ::onnx::TensorShapeProto;
    for(auto& dim : dims)
    {
        ts->add_dim()->set_dim_value(dim);
    }
    tt->set_elem_type(type);
    tt->set_allocated_shape(ts);
    tp->set_allocated_tensor_type(tt);
    return tp;
}

void fill_node_proto(node nd, ::onnx::NodeProto* nd_proto)
{
    auto op_name = *nd.get_attr("op_name").data<std::string>();
    auto cvt = export_registry::create(op_name);
    if(!cvt)
    {
        throw std::runtime_error(op_name + " operator is not support.");
    }
    cvt->convert(nd, nd_proto, nullptr);
    for (auto& nd_in : nd.get_inputs())
    {
        auto name = nd_in.get_name();
        auto onnx_in = nd_proto->add_input();
        onnx_in->assign(name.begin(), name.end());
    }
    {
        auto name = nd.get_name();
        auto onnx_out = nd_proto->add_output();
        onnx_out->assign(name.begin(), name.end());
    }
}

template <>
std::unique_ptr<std::vector<int>>
get_node_proto_attr<std::vector<int>, AttributeProto_AttributeType_INTS>(
    const ::onnx::NodeProto* nd_proto, std::string name)
{
    std::vector<int> vals;
    for(auto& attr : nd_proto->attribute())
    {
        if(attr.name() == name)
        {
            for(auto& v : attr.ints())
            {
                vals.push_back(v);
            }
            return std::make_unique<std::vector<int>>(std::vector<int>(vals));
        }
    }
    return nullptr;
}

template <>
std::unique_ptr<int>
get_node_proto_attr<int, AttributeProto_AttributeType_INT>(
    const ::onnx::NodeProto* nd_proto, std::string name)
{
    for(auto& attr : nd_proto->attribute())
    {
        if(attr.name() == name)
        {
            return std::make_unique<int>(int(attr.i()));
        }
    }
    return nullptr;
}

template <>
std::unique_ptr<std::string>
get_node_proto_attr<std::string, AttributeProto_AttributeType_STRING>(
    const ::onnx::NodeProto* nd_proto, std::string name)
{
    for(auto& attr : nd_proto->attribute())
    {
        if(attr.name() == name)
        {
            return std::make_unique<std::string>(std::string(attr.s()));
        }
    }
    return nullptr;
}

type::TypeInfo get_type_byte_size(int32_t enum_type)
{
    type::TypeInfo ti = type::float32();
    switch (enum_type)
    {
    case TensorProto::DataType::TensorProto_DataType_FLOAT:
    {
        ti = type::float32();
        break;
    }
    case TensorProto::DataType::TensorProto_DataType_INT32:
    {
        ti = type::int32();
        break;
    }
    case TensorProto::DataType::TensorProto_DataType_INT64:
    {
        ti = type::int64();
        break;
    }
    default:
    {
        throw std::runtime_error(std::to_string(enum_type) + " type not suppports.");
    }
    }

    return ti;
}

} // end namesapce onnx
} // end namespace mlfe