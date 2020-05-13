#include "activations.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/tensor.h"
#include "mlfe/core/gradient_helper.h"

namespace mlfe{ namespace functional{

class ReLUGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_context().get_input(0);
        Tensor dx = create_variable(x.shape());
        OpAlgoContext cxt("ReLUGradient");
        cxt.add_input(x);
        cxt.add_input(y);
        cxt.add_input(dy);
        cxt.add_output(dx);
        dx.set_context(cxt);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReLU, ReLUGradient)

class SigmoidGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        Tensor x = ctx_y.get_input(0);
        Tensor dx = create_variable(x.shape());
        OpAlgoContext ctx_dx("SigmoidGradient");
        ctx_dx.add_input(x);
        ctx_dx.add_input(y);
        ctx_dx.add_input(dy);
        ctx_dx.add_output(dx);
        dx.set_context(ctx_dx);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Sigmoid, SigmoidGradient)

Tensor relu(Tensor x){
    Tensor y = functional::create_variable(x.shape());
    OpAlgoContext ctx("ReLU");
    ctx.add_input(x);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor sigmoid(Tensor x){
    Tensor y = functional::create_variable(x.shape());
    OpAlgoContext ctx("Sigmoid");
    ctx.add_input(x);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
