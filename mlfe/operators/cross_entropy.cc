#include "cross_entropy.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{

class SigmoidXEntropyGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor logit = y.get_children()[0];
        Tensor label = y.get_children()[1];
        Tensor logit_grad = functional::create_variable(logit.shape());
        OpAlgoContext ctx("SigmoidCrossEntropyGradient");
        logit_grad.add_child(logit);
        logit_grad.add_child(label);
        logit_grad.add_child(y);
        logit_grad.add_child(dy);
        Tensor::AssignOpFunctor(logit_grad, ctx);
        in_grads.push_back(logit_grad);
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SigmoidCrossEntropy, SigmoidXEntropyGradient)

class SoftmaxCrossEntropyWithLabelGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor logit = y.get_children()[0];
        Tensor label = y.get_children()[1];
        Tensor logit_grad = functional::create_variable(logit.shape());
        OpAlgoContext ctx("SoftmaxCrossEntropyWithLabelGradient");
        logit_grad.add_child(logit);
        logit_grad.add_child(label);
        logit_grad.add_child(y);
        logit_grad.add_child(dy);
        Tensor::AssignOpFunctor(logit_grad, ctx);
        in_grads.push_back(logit_grad);
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SoftmaxCrossEntropyWithLabel, SoftmaxCrossEntropyWithLabelGradient)

namespace functional{

Tensor softmax_cross_entropy(Tensor logit, Tensor label){
    Tensor xent = create_variable({logit.shape()[0]});
    OpAlgoContext ctx("SoftmaxCrossEntropyWithLabel");
    xent.add_child(logit);
    xent.add_child(label);
    Tensor::AssignOpFunctor(xent, ctx);

    return xent;
}

Tensor sigmoid_cross_entropy(Tensor logit, Tensor label){
    Tensor xent = create_variable({logit.shape()[0]});
    OpAlgoContext ctx("SigmoidCrossEntropy");
    xent.add_child(logit);
    xent.add_child(label);
    Tensor::AssignOpFunctor(xent, ctx);

    return xent;
}

} // end namespace functional
} // end namespace mlfe
