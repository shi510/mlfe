#include "mlfe/utils/onnx/onnx_registry.h"
#include "mlfe/utils/onnx/onnx_helper.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/operators/convolution_utils.h"
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

// conv
class conv_op : public import_impl<conv_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto& w = inputs[nd_proto.input()[1]];
        auto kshape = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "kernel_shape");
        auto strides = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "strides");
        auto auto_pad = get_node_proto_attr<std::string,
            ::onnx::AttributeProto_AttributeType_STRING>(&nd_proto, "auto_pad");
        auto group = get_node_proto_attr<int,
            ::onnx::AttributeProto_AttributeType_INT>(&nd_proto, "group");
        auto dilations = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "dilations");
        auto pads = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "pads");
        bool same_out = false;
        if(auto_pad){
            pads = std::make_unique<std::vector<int>>();
            if(*auto_pad == "SAME_UPPER" || *auto_pad == "SAME_LOWER"){
                int in_h = x.shape()[2];
                int in_w = x.shape()[3];
                if(get_data_order_prefer() == data_order::nhwc){
                    in_h = x.shape()[1];
                    in_w = x.shape()[2];
                }
                int pad_h = util::calc_conv2d_pad_size_for_same_output(
                    in_h, w.shape()[2], strides->at(0));
                int pad_w = util::calc_conv2d_pad_size_for_same_output(
                    in_w, w.shape()[3], strides->at(1));
                pads->push_back(pad_h);
                pads->push_back(pad_w);
            }
            else{
                pads->push_back(0);
                pads->push_back(0);
            }
        }
        Tensor y;
        assert(dilations->size() == 2);
        assert(dilations->at(0) == 1);
        assert(dilations->at(1) == 1);
        if(*group == 1){
            Tensor tw = w;
            if(get_data_order_prefer() == data_order::nhwc){
                auto wo = w.shape()[0];
                auto wi = w.shape()[1];
                auto wh = w.shape()[2];
                auto ww = w.shape()[3];
                tw = functional::create_variable({wo, wh, ww, wi});
                math::transpose<float, CPUContext>(
                    w.data<float>(),
                    w.shape(),
                    {0, 2, 3, 1},
                    tw.mutable_data<float>()
                    );
            }
            y = functional::conv2d(x, tw, *strides, *pads);
            inputs[nd_proto.input()[1]] = tw;
        }
        else if(*group == w.shape()[0] && w.shape()[1] == 1){
            Tensor tw = w;
            if(get_data_order_prefer() == data_order::nhwc){
                auto wo = w.shape()[0];
                auto wi = w.shape()[1];
                auto wh = w.shape()[2];
                auto ww = w.shape()[3];
                tw = functional::create_variable({wi, wh, ww, wo});
                math::transpose<float, CPUContext>(
                    w.data<float>(),
                    w.shape(),
                    {1, 2, 3, 0},
                    tw.mutable_data<float>()
                    );
                tw.reshape({tw.shape()[1], tw.shape()[2], tw.shape()[3]});
            }
            y = functional::depthwise_conv2d(x, tw, *strides, *pads);
            inputs[nd_proto.input()[1]] = tw;
        }
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool conv_op::__is_registered = conv_op::regist("Conv");

// maxpool
class maxpool_op : public import_impl<maxpool_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto kshape = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "kernel_shape");
        auto strides = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "strides");
        auto pads = get_node_proto_attr<std::vector<int>,
            ::onnx::AttributeProto_AttributeType_INTS>(&nd_proto, "pads");
        auto auto_pad = get_node_proto_attr<std::string,
            ::onnx::AttributeProto_AttributeType_STRING>(&nd_proto, "auto_pad");
        bool same_out = false;
        if(auto_pad){
            pads = std::make_unique<std::vector<int>>();
            if(*auto_pad == "SAME_UPPER" || *auto_pad == "SAME_LOWER"){
                int in_h = x.shape()[2];
                int in_w = x.shape()[3];
                if(get_data_order_prefer() == data_order::nhwc){
                    in_h = x.shape()[1];
                    in_w = x.shape()[2];
                }
                int pad_h = util::calc_conv2d_pad_size_for_same_output(
                    in_h, kshape->at(0), strides->at(0));
                int pad_w = util::calc_conv2d_pad_size_for_same_output(
                    in_w, kshape->at(1), strides->at(1));
                pads->push_back(pad_h);
                pads->push_back(pad_w);
            }
            else{
                pads->push_back(0);
                pads->push_back(0);
            }
        }
        auto y = functional::pool_max(x, *kshape, *strides, { 0, 0 });
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool maxpool_op::__is_registered = maxpool_op::regist("MaxPool");

// global average pool
class gavg_pool2d_op : public import_impl<gavg_pool2d_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto y = functional::global_average_pool(x);
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool gavg_pool2d_op::__is_registered = gavg_pool2d_op::regist("GlobalAveragePool");

// batchnormalize
class batchnorm_op : public import_impl<batchnorm_op> {
public:
    void convert(const ::onnx::NodeProto& nd_proto,
        std::map<std::string, Tensor>& inputs) override
    {
        auto& x = inputs[nd_proto.input()[0]];
        auto& scales = inputs[nd_proto.input()[1]];
        auto& biases = inputs[nd_proto.input()[2]];
        auto& mean = inputs[nd_proto.input()[3]];
        auto& var = inputs[nd_proto.input()[4]];
        auto eps = get_node_proto_attr<float,
            ::onnx::AttributeProto_AttributeType_FLOAT>(&nd_proto, "epsilon");
        auto y = functional::batch_normalize(x, scales, biases, mean, var, *eps);
        inputs[nd_proto.output()[0]] = y;
    }

private:
    static bool __is_registered;
};

bool batchnorm_op::__is_registered = batchnorm_op::regist("BatchNormalization");

} // end namespace onnx_import
} // end namesapce onnx
} // end namespace mlfe