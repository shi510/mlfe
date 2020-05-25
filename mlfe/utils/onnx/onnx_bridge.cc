#include "mlfe/utils/onnx/onnx_bridge.h"
#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/utils/onnx/onnx_registry.h"
#include "mlfe/core/tensor.h"
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

} // end namesapce util
} // end namespace mlfe