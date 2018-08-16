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
        auto x = odc->Input("X");
        auto loss = odc->Output("Loss");
        loss.Reshape({ x.Shape()[0] }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SigmoidCrossEntropy)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class SigmoidXEntropyGradient : public GradientHelper{
public:
    SigmoidXEntropyGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor t = odc->Input("Target");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("SigmoidCrossEntropyGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple("Target", t))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(SigmoidCrossEntropy, SigmoidXEntropyGradient)

REGIST_OP(SoftmaxCrossEntropyWithLabel)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Output("Loss", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto loss = odc->Output("Loss");
        loss.Reshape({ x.Shape()[0] }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SoftmaxCrossEntropyWithLabel)
    .Input("X", "float32")
    .Input("Target", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class SoftmaxCrossEntropyWithLabelGradient : public GradientHelper{
public:
    SoftmaxCrossEntropyWithLabelGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor t = odc->Input("Target");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("SoftmaxCrossEntropyWithLabelGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple("Target", t))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(SoftmaxCrossEntropyWithLabel, SoftmaxCrossEntropyWithLabelGradient)

namespace functional{

Tensor SigmoidCrossEntropy(Tensor x, Tensor y){
    Tensor loss;

    auto dep = OpDependency::Builder("SigmoidCrossEntropy")
        .Input(std::make_tuple("X", x))
        .Input(std::make_tuple("Target", y))
        .Output(std::make_tuple("Loss", loss))
        .Finish();

    loss = Tensor::DependencyAdder(dep);

    return loss;
}

Tensor SoftmaxCrossEntropy(Tensor x, Tensor y){
    Tensor loss;

    auto dep = OpDependency::Builder("SoftmaxCrossEntropyWithLabel")
        .Input(std::make_tuple("X", x))
        .Input(std::make_tuple("Target", y))
        .Output(std::make_tuple("Loss", loss))
        .Finish();

    loss = Tensor::DependencyAdder(dep);

    return loss;
}

} // end namespace functional
} // end namespace mlfe
