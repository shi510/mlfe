#include "convolution.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

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
    auto x_shape = x.shape();
    auto w_shape = w.shape();
    int out_h = (x_shape[2] - w_shape[2] + 2 * pads[0]) / strides[0] + 1;
    int out_w = (x_shape[3] - w_shape[3] + 2 * pads[1]) / strides[1] + 1;
    Tensor y = create_variable({x_shape[0], w_shape[0], out_h, out_w});
    OpAlgoContext ctx("Convolution");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", pads});
    ctx.add_input(x);
    ctx.add_input(w);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
