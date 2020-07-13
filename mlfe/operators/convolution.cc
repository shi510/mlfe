#include "convolution.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"
#include "mlfe/operators/convolution_utils.h"
#include <cmath>

namespace mlfe{ namespace functional{

class ConvolutionGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        VecTensor in_grads;
        auto ctx = y.get_context();
        Tensor x = ctx.get_input(0);
        Tensor w = ctx.get_input(1);
        Tensor dx = functional::create_variable(x.shape());
        Tensor dw = functional::create_variable(w.shape());
        OpAlgoContext ctx_x_grad("Conv2DGradientInputGradient");
        OpAlgoContext ctx_w_grad("Conv2DGradientFilterGradient");
        ctx_x_grad.add_attr({ "strides", ctx.get_attr<IntVec>("strides") });
        ctx_x_grad.add_attr({ "pads", ctx.get_attr<IntVec>("pads") });
        ctx_x_grad.add_input(w);
        ctx_x_grad.add_input(dy);
        ctx_x_grad.add_output(dx);
        ctx_w_grad.add_attr({ "strides", ctx.get_attr<IntVec>("strides") });
        ctx_w_grad.add_attr({ "pads", ctx.get_attr<IntVec>("pads") });
        ctx_w_grad.add_input(x);
        ctx_w_grad.add_input(dy);
        ctx_w_grad.add_output(dw);
        dx.set_context(ctx_x_grad);
        dw.set_context(ctx_w_grad);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        w.set_backprop_node(dw.get_node());
        w.set_gradient(dw);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Convolution, ConvolutionGradient)

Tensor conv2d(Tensor x,
              Tensor w,
              std::vector<type::int32::T> strides,
              std::vector<type::int32::T> pads
              ){
    Tensor y;
    OpAlgoContext ctx("Convolution");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", pads});
    ctx.add_input(x);
    ctx.add_input(w);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor conv2d(Tensor x,
    Tensor w,
    std::vector<int32_t> strides,
    bool same_out)
{
    Tensor y;
    int32_t kernel_h = w.shape()[2];
    int32_t kernel_w = w.shape()[3];
    int32_t input_h = x.shape()[2];
    int32_t input_w = x.shape()[3];
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    if(get_data_order_prefer() == data_order::nhwc){
        kernel_h = w.shape()[1];
        kernel_w = w.shape()[2];
        input_h = x.shape()[1];
        input_h = x.shape()[2];
    }
    if(same_out){
        pad_h = util::calc_conv2d_pad_size_for_same_output(
            input_h, kernel_h, strides[0]
        );
        pad_w = util::calc_conv2d_pad_size_for_same_output(
            input_w, kernel_w, strides[1]
        );
    }
    OpAlgoContext ctx("Convolution");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", std::vector<int32_t>{pad_h, pad_w}});
    ctx.add_input(x);
    ctx.add_input(w);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor depthwise_conv2d(Tensor x,
    Tensor w,
    std::vector<int32_t> strides,
    std::vector<int32_t> pads
    ){
    Tensor y;
    OpAlgoContext ctx("DepthwiseConv2d");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", pads});
    ctx.add_input(x);
    ctx.add_input(w);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor depthwise_conv2d(Tensor x,
    Tensor w,
    std::vector<int32_t> strides,
    bool same_out)
{
    Tensor y;
    int32_t kernel_h = w.shape()[2];
    int32_t kernel_w = w.shape()[3];
    int32_t input_h = x.shape()[2];
    int32_t input_w = x.shape()[3];
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    if(get_data_order_prefer() == data_order::nhwc){
        kernel_h = w.shape()[1];
        kernel_w = w.shape()[2];
        input_h = x.shape()[1];
        input_h = x.shape()[2];
    }
    if(same_out){
        pad_h = util::calc_conv2d_pad_size_for_same_output(
            input_h, kernel_h, strides[0]
        );
        pad_w = util::calc_conv2d_pad_size_for_same_output(
            input_w, kernel_w, strides[1]
        );
    }
    OpAlgoContext ctx("DepthwiseConv2d");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", std::vector<int32_t>{pad_h, pad_w}});
    ctx.add_input(x);
    ctx.add_input(w);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
