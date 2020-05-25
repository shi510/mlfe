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

} // end namespace onnx_export
} // end namesapce onnx
} // end namespace mlfe