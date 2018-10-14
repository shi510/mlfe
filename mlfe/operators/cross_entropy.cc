#include "cross_entropy.h"
#include "../core/op_dep.h"
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

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = odc->Input(0);
        Tensor t = odc->Input(1);
        Tensor dx;

        dep = OpDependency::Builder("SigmoidCrossEntropyGradient")
            .Input(x)
            .Input(t)
            .Input(dy)
            .Output(dx)
            .Finish();

        gpair[x] = dx;

        return gpair;
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

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = odc->Input(0);
        Tensor t = odc->Input(1);
        Tensor dx;

        dep = OpDependency::Builder("SoftmaxCrossEntropyWithLabelGradient")
            .Input(x)
            .Input(t)
            .Input(dy)
            .Output(dx)
            .Finish();

        gpair[x] = dx;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(SoftmaxCrossEntropyWithLabel, SoftmaxCrossEntropyWithLabelGradient)

namespace functional{

Tensor SigmoidCrossEntropy(Tensor x, Tensor y){
    Tensor loss;

    auto dep = OpDependency::Builder("SigmoidCrossEntropy")
        .Input(x)
        .Input(y)
        .Output(loss)
        .Finish();

    loss = Tensor::DependencyAdder(dep);
    loss.add_child(x);
    loss.add_child(y);

    return loss;
}

Tensor SoftmaxCrossEntropy(Tensor x, Tensor y){
    Tensor loss;

    auto dep = OpDependency::Builder("SoftmaxCrossEntropyWithLabel")
        .Input(x)
        .Input(y)
        .Output(loss)
        .Finish();

    loss = Tensor::DependencyAdder(dep);

    loss.add_child(x);
    loss.add_child(y);

    return loss;
}

} // end namespace functional
} // end namespace mlfe
