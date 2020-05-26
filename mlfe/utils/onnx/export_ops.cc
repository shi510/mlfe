#include "mlfe/utils/onnx/onnx_registry.h"
#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/core/op_algo.h"

namespace mlfe{
namespace onnx {
namespace onnx_export {

// reshape
class rehspae_op : public export_impl<rehspae_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        nd_proto->set_op_type("Reshape");
        auto y = nd.get_attr("tensor").data<Tensor>();
        auto ctx = y->get_context();
        auto shape_t = ctx.get_input(1);
        auto onnx_input = g->add_input();
        auto in_tp = make_type_proto(shape_t.shape(),
            ::onnx::TensorProto_DataType_INT64);
        onnx_input->set_name(shape_t.get_node().get_name());
        onnx_input->set_allocated_type(in_tp);
        // add initializers for trainable variables.
        auto onnx_init = g->add_initializer();
        onnx_init->set_name(shape_t.get_node().get_name());
        onnx_init->set_data_type(::onnx::TensorProto_DataType_INT64);
        onnx_init->set_raw_data(shape_t.data<void>(), shape_t.size() * sizeof(int64_t));
        for (auto& i : shape_t.shape())
        {
            onnx_init->add_dims(i);
        }
    }

private:
    static bool __is_registered;
};

bool rehspae_op::__is_registered = rehspae_op::regist("Reshape");

// broadcast
class boardcast_op : public export_impl<boardcast_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        nd_proto->set_op_type("Broadcasting");
    }

private:
    static bool __is_registered;
};

bool boardcast_op::__is_registered = boardcast_op::regist("Broadcasting");

// add matrix-vector
class add_matvec_op : public export_impl<add_matvec_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override{
        nd_proto->set_op_type("Add");
    }

private:
    static bool __is_registered;
};

bool add_matvec_op::__is_registered = add_matvec_op::regist("MatrixVectorAdd");

// add with broadcasting
class add_broadcast_op : public export_impl<add_broadcast_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        nd_proto->set_op_type("Add");
    }

private:
    static bool __is_registered;
};

bool add_broadcast_op::__is_registered = add_broadcast_op::regist("AddWithBroadcast");

// sigmoid
class sigmoid_op : public export_impl<sigmoid_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override{
        nd_proto->set_op_type("Sigmoid");
    }

private:
    static bool __is_registered;
};

bool sigmoid_op::__is_registered = sigmoid_op::regist("Sigmoid");

// Relu
class relu_op : public export_impl<relu_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        nd_proto->set_op_type("Relu");
    }

private:
    static bool __is_registered;
};

bool relu_op::__is_registered = relu_op::regist("ReLU");

// matmul
class matmul_op : public export_impl<matmul_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override{
        nd_proto->set_op_type("MatMul");
    }

private:
    static bool __is_registered;
};

bool matmul_op::__is_registered = matmul_op::regist("MatMul");

// conv
class conv_op : public export_impl<conv_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override{
        auto var = nd.get_attr("tensor").data<Tensor>();
        auto ctx = var->get_context();
        auto wshape = ctx.get_input(1).shape();
        auto strides = ctx.get_attr<std::vector<int>>("strides");
        auto pads = ctx.get_attr<std::vector<int>>("pads");
        auto attr = nd_proto->add_attribute();
        attr->set_name("kernel_shape");
        attr->add_ints(wshape[2]);
        attr->add_ints(wshape[3]);

        attr = nd_proto->add_attribute();
        attr->set_name("strides");
        attr->add_ints(strides[0]);
        attr->add_ints(strides[1]);

        attr = nd_proto->add_attribute();
        attr->set_name("dilations");
        attr->add_ints(1);
        attr->add_ints(1);

        attr = nd_proto->add_attribute();
        attr->set_name("group");
        attr->set_i(1);

        attr = nd_proto->add_attribute();
        attr->set_name("auto_pad");
        if (ctx.get_input(0).shape()[2] == ctx.get_output(0).shape()[2] &&
            ctx.get_input(0).shape()[3] == ctx.get_output(0).shape()[3])
        {
            attr->set_s("SAME_UPPER");
        }
        else
        {
            attr->set_s("NOSET");
        }
        nd_proto->set_op_type("Conv");
    }

private:
    static bool __is_registered;
};

bool conv_op::__is_registered = conv_op::regist("Convolution");

// maxpool
class maxpool_op : public export_impl<maxpool_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        auto var = nd.get_attr("tensor").data<Tensor>();
        auto ctx = var->get_context();
        auto wshape = ctx.get_attr<std::vector<int>>("kernel");
        auto strides = ctx.get_attr<std::vector<int>>("stride");
        auto pads = ctx.get_attr<std::vector<int>>("padding");
        auto attr = nd_proto->add_attribute();
        attr->set_name("kernel_shape");
        attr->add_ints(wshape[0]);
        attr->add_ints(wshape[1]);

        attr = nd_proto->add_attribute();
        attr->set_name("strides");
        attr->add_ints(strides[0]);
        attr->add_ints(strides[1]);

        attr = nd_proto->add_attribute();
        attr->set_name("auto_pad");
        if (ctx.get_input(0).shape()[2] == ctx.get_output(0).shape()[2] &&
            ctx.get_input(0).shape()[3] == ctx.get_output(0).shape()[3])
        {
            attr->set_s("SAME_UPPER");
        }
        else
        {
            attr->set_s("NOSET");
        }
        nd_proto->set_op_type("MaxPool");
    }

private:
    static bool __is_registered;
};

bool maxpool_op::__is_registered = maxpool_op::regist("MaxPool");

// batchnorm
class batchnorm_op : public export_impl<batchnorm_op> {
public:
    void convert(node nd, ::onnx::NodeProto* nd_proto, ::onnx::GraphProto* g) override {
        auto y = nd.get_attr("tensor").data<Tensor>();
        auto ctx = y->get_context();
        auto running_mean = ctx.get_attr<Tensor>("running_mean");
        auto running_var = ctx.get_attr<Tensor>("running_var");

        {
            auto onnx_input = g->add_input();
            auto in_tp = make_type_proto(running_mean.shape(),
                ::onnx::TensorProto_DataType_FLOAT);
            onnx_input->set_name(running_mean.get_node().get_name());
            onnx_input->set_allocated_type(in_tp);

            auto onnx_init = g->add_initializer();
            onnx_init->set_name(running_mean.get_node().get_name());
            onnx_init->set_data_type(::onnx::TensorProto_DataType_FLOAT);
            onnx_init->set_raw_data(running_mean.data<void>(),
                running_mean.size() * sizeof(float));
            for (auto& i : running_mean.shape()){
                onnx_init->add_dims(i);
            }
            *nd_proto->add_input() = onnx_input->name();
        }

        {
            auto onnx_input = g->add_input();
            auto in_tp = make_type_proto(running_var.shape(),
                ::onnx::TensorProto_DataType_FLOAT);
            onnx_input->set_name(running_var.get_node().get_name());
            onnx_input->set_allocated_type(in_tp);

            auto onnx_init = g->add_initializer();
            onnx_init->set_name(running_var.get_node().get_name());
            onnx_init->set_data_type(::onnx::TensorProto_DataType_FLOAT);
            onnx_init->set_raw_data(running_var.data<void>(),
                running_var.size() * sizeof(float));
            for (auto& i : running_var.shape()){
                onnx_init->add_dims(i);
            }
            *nd_proto->add_input() = onnx_input->name();
        }
        nd_proto->set_op_type("BatchNormalization");
    }

private:
    static bool __is_registered;
};

bool batchnorm_op::__is_registered = batchnorm_op::regist("BatchNormSpatial");

} // end namespace onnx_export
} // end namesapce onnx
} // end namespace mlfe