#include "mlfe/utils/onnx/onnx_bridge.h"
#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/utils/onnx/onnx_registry.h"
#include "mlfe/core/device.h"
#include "mlfe/core/tensor.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <map>

namespace mlfe{
namespace onnx{

using namespace ::onnx;

bool export_onnx_model(module::model m, std::string file_name)
{
    using onnx::TensorProto_DataType;
    auto onnx_model{ std::make_unique<onnx::ModelProto>() };
    auto onnx_graph = new onnx::GraphProto;
    auto in = m.get_input();
    auto out = m.get_output();
    auto train_vars = m.get_train_variables();
    auto exec_list = topological_sort(out.get_node());
    // remove duplicated nodes.
    std::sort(exec_list.begin(), exec_list.end(), [](node a, node b){
        return a.get_name() > b.get_name();
        });
    exec_list.erase(std::unique(exec_list.begin(), exec_list.end()), exec_list.end());
    std::sort(exec_list.begin(), exec_list.end(), [](node a, node b){
        return a.get_order() < b.get_order();
        });
    exec_list.erase(std::remove_if(exec_list.begin(), exec_list.end(),
        [](node nd){
            auto op_name = *nd.get_attr("op_name").data<std::string>();
            return op_name == "Identity";
        }), exec_list.end());
    onnx_model->set_ir_version(::onnx::IR_VERSION);
    onnx_graph->set_name(m.get_name());
    auto onnx_input = onnx_graph->add_input();
    auto in_tp = make_type_proto(in.shape(), TensorProto_DataType_FLOAT);
    onnx_input->set_name(in.get_node().get_name());
    onnx_input->set_allocated_type(in_tp);

    auto onnx_output = onnx_graph->add_output();
    auto out_tp = make_type_proto(out.shape(), TensorProto_DataType_FLOAT);
    onnx_output->set_name(out.get_node().get_name());
    onnx_output->set_allocated_type(out_tp);
    for(auto& nd : exec_list)
    {
        auto nd_proto = onnx_graph->add_node();
        //fill_node_proto(nd, nd_proto);
        auto op_name = *nd.get_attr("op_name").data<std::string>();
        auto t = *nd.get_attr("tensor").data<Tensor>();
        auto cvt = export_registry::create(op_name);
        if (!cvt)
        {
            throw std::runtime_error(op_name + " operator is not support.");
        }
        
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
        auto vi = onnx_graph->add_value_info();
        vi->set_allocated_type(
            make_type_proto(t.shape(), TensorProto_DataType_FLOAT));
        vi->set_name(t.get_node().get_name());
        cvt->convert(nd, nd_proto, onnx_graph);
    }
    for(auto& var : train_vars)
    {
        // add inputs for trainable variables.
        auto onnx_input = onnx_graph->add_input();
        auto in_tp = make_type_proto(var.shape(),
            TensorProto_DataType_FLOAT);
        onnx_input->set_name(var.get_node().get_name());
        onnx_input->set_allocated_type(in_tp);
        
        // add initializers for trainable variables.
        auto onnx_init = onnx_graph->add_initializer();
        onnx_init->set_name(var.get_node().get_name());
        onnx_init->set_data_type(TensorProto_DataType_FLOAT);
        onnx_init->set_raw_data(var.data<void>(), var.size() * sizeof(float));
        for(auto& i : var.shape())
        {
            onnx_init->add_dims(i);
        }
    }
    onnx_model->set_allocated_graph(onnx_graph);
    onnx_model->add_opset_import()->set_version(IR_VERSION);
    auto f = std::ofstream(file_name,
        std::ios_base::binary | std::ios_base::trunc);
    onnx_model->SerializeToOstream(&f);
    f.close();
    return true;
}

module::model import_onnx_model(std::string file_name)
{
    Tensor model_in;
    Tensor model_out;
    std::map<std::string, Tensor> collector;
    auto onnx_model{ std::make_unique<onnx::ModelProto>() };
    if(!read_onnx_model(file_name, onnx_model.get()))
    {
        return module::model(model_in, model_out, "");
    }
    for(auto& in : onnx_model->graph().input())
    {
        std::vector<int32_t> dims;
        if(in.type().has_tensor_type())
        {
            if(in.type().tensor_type().has_shape())
            {
                auto in_dims = in.type().tensor_type().shape().dim();
                for(auto& d : in_dims)
                {
                    dims.push_back(d.dim_value());
                }
                auto ti = get_type_byte_size(in.type().tensor_type().elem_type());
                auto t = functional::create_variable(dims, ti);
                t.get_node().set_name(in.name());
                collector[in.name()] = t;
            }
            else
            {
                collector[in.name()] = Tensor();
            }
        }
        else 
        {
            collector[in.name()] = Tensor();
        }
    }
    for(auto& out : onnx_model->graph().output())
    {
        std::vector<int32_t> dims;
        if(out.type().has_tensor_type())
        {
            if(out.type().tensor_type().has_shape())
            {
                auto out_dims = out.type().tensor_type().shape().dim();
                for(auto& d : out_dims)
                {
                    dims.push_back(d.dim_value());
                }
                auto t = functional::create_variable(dims);
                t.get_node().set_name(out.name());
                collector[out.name()] = t;
            }
            else
            {
                collector[out.name()] = Tensor();
            }
        }
        else
        {
            collector[out.name()] = Tensor();
        }
    }
    for(auto& info : onnx_model->graph().initializer())
    {
        std::string name = info.name();
        std::vector<int32_t> dims;
        int dim_size = 0;
        int byte_size = 0;
        const void* ptr = nullptr;
        
        for(auto& dim : info.dims())
        {
            dims.push_back(dim);
        }
        dim_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        switch (info.data_type())
        {
        case TensorProto::DataType::TensorProto_DataType_FLOAT:
        {
            if(info.float_data_size() != 0)
            {
                ptr = reinterpret_cast<const void*>(info.float_data().data());
            }
            byte_size = dim_size * sizeof(float);
            break;
        }
        case TensorProto::DataType::TensorProto_DataType_INT32:
        {
            if(info.int32_data_size() != 0)
            {
                ptr = reinterpret_cast<const void*>(info.int32_data().data());
            }
            byte_size = dim_size * sizeof(int32_t);
            break;
        }
        case TensorProto::DataType::TensorProto_DataType_INT64:
        {
            if(info.int64_data_size() != 0)
            {
                ptr = reinterpret_cast<const void*>(info.int64_data().data());
            }
            byte_size = dim_size * sizeof(int64_t);
            break;
        }

        default:
        {
            auto type_name = info.DataType_Name(info.data_type());
            throw std::runtime_error(type_name +
                " type of initializer not suppports.");
        }
        } // end of switch
        if(collector.find(info.name()) == collector.end())
        {
            throw std::runtime_error(info.name() +
                " intializer can not find in graph inputs.");
        }
        if(!ptr)
        {
            ptr = reinterpret_cast<const void*>(info.raw_data().data());
        }
        auto& t = collector[info.name()];
        std::copy(reinterpret_cast<const uint8_t *>(ptr),
            reinterpret_cast<const uint8_t*>(ptr) + byte_size, t.begin<uint8_t>());
    } // end for-loop

    model_in = collector[onnx_model->graph().input()[0].name()];
    if(get_data_order_prefer() == data_order::nhwc){
        auto N = model_in.shape()[0];
        auto C = model_in.shape()[1];
        auto H = model_in.shape()[2];
        auto W = model_in.shape()[3];
        model_in.reshape({N, H, W, C});
    }
    collector[onnx_model->graph().input()[0].name()] = model_in;

    for(auto& nd_proto : onnx_model->graph().node())
    {
        std::string op_name = nd_proto.op_type();
        auto cvt = import_registry::create(op_name);
        if(!cvt)
        {
            throw std::runtime_error(op_name + " operator is not support.");
        }
        cvt->convert(nd_proto, collector);
    }
    model_out = collector[onnx_model->graph().output()[0].name()];
    return module::model(model_in, model_out, onnx_model->graph().name());
}

} // end namesapce util
} // end namespace mlfe