#include "reduce.h"
#include "../core/op_dep.h"
#include "../core/tensor.h"
#include "../core/op_design.h"
#include "../core/gradient_helper.h"

namespace mlfe{

REGIST_OP(ReduceMean)
    .Input("X", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto y = odc->Output(0);
        y.Reshape({ 1 }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(ReduceMean)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ReduceMeanGradient : public GradientHelper{
public:
    ReduceMeanGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = odc->Input(0);
        Tensor dx;

        dep = OpDependency::Builder("ReduceMeanGradient")
            .Input(x)
            .Input(dy)
            .Output(dx)
            .Finish();

        gpair[x] = dx;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor Mean(Tensor x){
    Tensor y;
    auto dep = OpDependency::Builder("ReduceMean")
        .Input(x)
        .Output(y)
        .Finish();

    y = Tensor::DependencyAdder(dep);
    y.add_child(x);
    return y;
}

} // end namespace functional
} // end namespace mlfe
