#include "mlfe/utils/onnx/onnx_registry.h"
#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/operators.h"

namespace mlfe {
namespace onnx {
namespace onnx_import {

// reshape
class reshape_op : public import_impl<reshape_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        if (nd_proto.input().size() == 2)
        {
            auto& x = inputs[nd_proto.input()[0]];
            auto& shape_t = inputs[nd_proto.input()[1]];
            std::vector<int> shape;
            shape.push_back(shape_t.data<int64_t>()[0]);
            shape.push_back(shape_t.data<int64_t>()[1]);
            auto y = functional::reshape(x, shape);
            inputs[nd_proto.output()[0]] = y;
        }
        else if (nd_proto.input().size() == 1)
        {
            assert(nd_proto.attribute().size() == 1);
            auto& x = inputs[nd_proto.input()[0]];
            auto shape_attr = nd_proto.attribute()[0].ints();
            std::vector<int> shape;
            for (int n = 0; n < shape_attr.size(); ++n)
            {
                shape.push_back(shape_attr[n]);
            }
            auto y = functional::reshape(x, shape);
            inputs[nd_proto.output()[0]] = y;
        }
    }

private:
    static bool __is_registered;
};

bool reshape_op::__is_registered = reshape_op::regist("Reshape");

// add
class add_op : public import_impl<add_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        if(nd_proto.input_size() == 2)
        {
            auto& x1 = inputs[nd_proto.input()[0]];
            auto& x2 = inputs[nd_proto.input()[1]];
            auto y = functional::add(x1, x2);
            inputs[nd_proto.output()[0]] = y;
        }
        else
        {
            std::vector<Tensor> xs;
            for(auto& in : nd_proto.input())
            {
                xs.push_back(inputs[in]);
            }
            auto y = functional::add_n(xs);
            inputs[nd_proto.output()[0]] = y;
        }
        
    }

private:
    static bool __is_registered;
};

bool add_op::__is_registered = add_op::regist("Add");

// sigmoid
class sigmoid_op : public import_impl<sigmoid_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto y = functional::sigmoid(x);
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool sigmoid_op::__is_registered = sigmoid_op::regist("Sigmoid");

// relu
class relu_op : public import_impl<relu_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto y = functional::relu(x);
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool relu_op::__is_registered = relu_op::regist("Relu");

// matmul
class matmul_op : public import_impl<matmul_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& a = inputs[nd_proto.input()[0]];
        auto& b = inputs[nd_proto.input()[1]];
        auto y = functional::matmul(a, b);
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool matmul_op::__is_registered = matmul_op::regist("MatMul");

} // end namespace onnx_import
} // end namesapce onnx
} // end namespace mlfe