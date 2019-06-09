#include "convolution.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

class ConvolutionGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor w = y.get_children()[1];
        Tensor dx = functional::create_variable(x.shape());
        Tensor dw = functional::create_variable(w.shape());
        OpAlgoContext ctx_x_grad("Conv2DGradientInputGradient");
        OpAlgoContext ctx_w_grad("Conv2DGradientFilterGradient");
        OpAlgoContext ctx = y.get_context();
        ctx_x_grad.add_attr({"strides", ctx.get_attr<IntVec>("strides")});
        ctx_x_grad.add_attr({"pads", ctx.get_attr<IntVec>("pads")});
        ctx_w_grad.add_attr({"strides", ctx.get_attr<IntVec>("strides")});
        ctx_w_grad.add_attr({"pads", ctx.get_attr<IntVec>("pads")});
        dx.add_child(w);
        dx.add_child(dy);
        dw.add_child(x);
        dw.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx_x_grad);
        Tensor::AssignOpFunctor(dw, ctx_w_grad);
        in_grads.push_back(dx);
        in_grads.push_back(dw);
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
    y.add_child(x);
    y.add_child(w);
    Tensor::AssignOpFunctor(y, ctx);
    
    return y;
}

} // end namespace functional
} // end namespace mlfe
