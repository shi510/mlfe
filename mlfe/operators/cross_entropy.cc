#include "cross_entropy.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{

REGIST_OP(SigmoidCrossEntropy)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Output("Loss", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto loss = odc->Output(0);
        loss.Reshape({ x.Shape()[0] }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SigmoidCrossEntropy)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class SigmoidXEntropyGradient : public GradientHelper{
public:
    SigmoidXEntropyGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor logit = y.get_children()[0];
        Tensor label = y.get_children()[1];
        Tensor logit_grad = functional::create_variable(logit.Shape());
        Tensor label_grad = functional::create_variable(label.Shape());
        OpAlgoContext ctx("SigmoidCrossEntropyGradient");
        logit_grad.add_child(logit);
        logit_grad.add_child(label);
        logit_grad.add_child(y);
        logit_grad.add_child(dy);
        label_grad.add_child(dy);
        Tensor::AssignOpFunctor(logit_grad, ctx);
        in_grads.push_back(logit_grad);
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SigmoidCrossEntropy, SigmoidXEntropyGradient)

REGIST_OP(SoftmaxCrossEntropyWithLabel)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Output("Loss", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto loss = odc->Output(0);
        loss.Reshape({ x.Shape()[0] }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SoftmaxCrossEntropyWithLabel)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class SoftmaxCrossEntropyWithLabelGradient : public GradientHelper{
public:
    SoftmaxCrossEntropyWithLabelGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor logit = y.get_children()[0];
        Tensor label = y.get_children()[1];
        Tensor logit_grad = functional::create_variable(logit.Shape());
        Tensor label_grad = functional::create_variable(label.Shape());
        OpAlgoContext ctx("SoftmaxCrossEntropyWithLabelGradient");
        logit_grad.add_child(logit);
        logit_grad.add_child(label);
        logit_grad.add_child(y);
        logit_grad.add_child(dy);
        label_grad.add_child(dy);
        Tensor::AssignOpFunctor(logit_grad, ctx);
        in_grads.push_back(logit_grad);
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SoftmaxCrossEntropyWithLabel, SoftmaxCrossEntropyWithLabelGradient)

namespace functional{

Tensor softmax_cross_entropy(Tensor logit, Tensor label){
    Tensor xent = create_variable({logit.Shape()[0]});
    OpAlgoContext ctx("SoftmaxCrossEntropyWithLabel");
    xent.add_child(logit);
    xent.add_child(label);
    Tensor::AssignOpFunctor(xent, ctx);

    return xent;
}

Tensor sigmoid_cross_entropy(Tensor logit, Tensor label){
    Tensor xent = create_variable({logit.Shape()[0]});
    OpAlgoContext ctx("SigmoidCrossEntropy");
    xent.add_child(logit);
    xent.add_child(label);
    Tensor::AssignOpFunctor(xent, ctx);

    return xent;
}

} // end namespace functional
} // end namespace mlfe
