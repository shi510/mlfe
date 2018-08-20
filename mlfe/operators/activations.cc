#include "activations.h"
#include "../core/op_dep.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(ReLU)
    .Input("X", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x = odc->Input("X");
        auto &y = odc->Output("Y");
        if (x.Name() != y.Name()){
            y.Reshape(x.Shape(), x.Type());
        }
    })
    .Finish();

REGIST_OP_GRAD(ReLU)
    .Input("X", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x = odc->Input("X");
        auto &dx = odc->Output("dX");
        dx.Reshape(x.Shape(), x.Type());
    })
    .Finish();

class ReLUGradient : public GradientHelper{
public:
    ReLUGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor y = odc->Output("Y");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("ReLUGradient")
            .Input({ "X", x })
            .Input({ "Y", y })
            .Input({ Gradient("Y"), dy })
            .Output({ Gradient("X"), dx })
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return{ dx, pairs };
    }
};

REGIST_GRADIENT_HELPER(ReLU, ReLUGradient)

REGIST_OP(Sigmoid)
    .Input("X", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x = odc->Input("X");
        auto &y = odc->Output("Y");
        if(x.Name() != y.Name()){
            y.Reshape(x.Shape(), x.Type());
        }
    })
    .Finish();

REGIST_OP_GRAD(Sigmoid)
    .Input("X", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x = odc->Input("X");
        auto &dx = odc->Output("dX");
        dx.Reshape(x.Shape(), x.Type());
    })
    .Finish();

class SigmoidGradient : public GradientHelper{
public:
    SigmoidGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor y = odc->Output("Y");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("SigmoidGradient")
            .Input({ "X", x })
            .Input({ "Y", y })
            .Input({ Gradient("Y"), dy })
            .Output({ Gradient("X"), dx })
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return{ dx, pairs };
    }
};

REGIST_GRADIENT_HELPER(Sigmoid, SigmoidGradient)

Tensor ReLU(Tensor x){
    Tensor y;

    auto dep = OpDependency::Builder("ReLU")
        .Input({ "X", x })
        .Output({ "Y", y })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

Tensor Sigmoid(Tensor x){
    Tensor y;

    auto dep = OpDependency::Builder("Sigmoid")
        .Input({ "X", x })
        .Output({ "Y", y })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

} // end namespace functional
} // end namespace mlfe
