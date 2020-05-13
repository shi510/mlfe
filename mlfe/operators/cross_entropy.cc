#include "cross_entropy.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{

class SigmoidXEntropyGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        Tensor logit = ctx_y.get_input(0);
        Tensor label = ctx_y.get_input(1);
        Tensor logit_grad = functional::create_variable(logit.shape());
        Tensor label_grad = functional::create_variable(label.shape());
        OpAlgoContext ctx_logit_grad("SigmoidCrossEntropyGradient");
        ctx_logit_grad.add_input(logit);
        ctx_logit_grad.add_input(label);
        ctx_logit_grad.add_input(y);
        ctx_logit_grad.add_input(dy);
        ctx_logit_grad.add_output(logit_grad);
        logit_grad.set_context(ctx_logit_grad);
        logit.set_backprop_node(logit_grad.get_node());
        logit.set_gradient(logit_grad);
        label_grad.set_backprop_node(dy.get_node());
        label_grad.set_gradient(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SigmoidCrossEntropy, SigmoidXEntropyGradient)

class SoftmaxCrossEntropyWithLabelGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        Tensor logit = ctx_y.get_input(0);
        Tensor label = ctx_y.get_input(1);
        Tensor logit_grad = functional::create_variable(logit.shape());
        OpAlgoContext ctx("SoftmaxCrossEntropyWithLabelGradient");
        ctx.add_input(dy);
        ctx.add_input(logit);
        ctx.add_input(label);
        ctx.add_input(y);
        ctx.add_output(logit_grad);
        logit_grad.set_context(ctx);
        logit.set_backprop_node(logit_grad.get_node());
        label.set_backprop_node(dy.get_node());
        logit.set_gradient(logit_grad);
        label.set_gradient(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SoftmaxCrossEntropyWithLabel, SoftmaxCrossEntropyWithLabelGradient)

namespace functional{

Tensor softmax_cross_entropy(Tensor logit, Tensor label){
    Tensor y = create_variable({ logit.shape()[0] });
    OpAlgoContext ctx("SoftmaxCrossEntropyWithLabel");
    ctx.add_input(logit);
    ctx.add_input(label);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor sigmoid_cross_entropy(Tensor logit, Tensor label){
    Tensor xent = create_variable({logit.shape()[0]});
    OpAlgoContext ctx("SigmoidCrossEntropy");
    ctx.add_input(logit);
    ctx.add_input(label);
    ctx.add_output(xent);
    xent.set_context(ctx);
    return xent;
}

} // end namespace functional
} // end namespace mlfe
