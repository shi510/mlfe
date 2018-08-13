#include "reduce.h"
#include "../core/op_dep.h"
#include "../core/tensor.h"
#include "../core/op_design.h"
#include "../core/gradient_helper.h"

namespace mlfe{

REGIST_OP(ReduceMean)
    .Input("X", "float32")
    .Input("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto y = odc->Output("Y");
        y.Reshape({ 1 }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(ReduceMean)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ReduceMeanGradient : public GradientHelper{
public:
    ReduceMeanGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("ReduceMeanGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor Mean(Tensor x){
    Tensor y;
    auto dep = OpDependency::Builder("ReduceMean")
        .Input(std::make_tuple("X", x))
        .Output(std::make_tuple("Y", y))
        .Finish();

    y = Tensor::DependencyAdder(dep);
    return y;
}

} // end namespace functional
} // end namespace mlfe
