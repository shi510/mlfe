#include "reduce.h"
#include "../core/op_algo.h"
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
        Tensor x = y.get_children()[0];
        Tensor dx = functional::variable(x.Shape());
        OpAlgoContext ctx("ReduceMeanGradient");
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx);

        gpair[x] = dx;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor mean(Tensor x){
    Tensor y = functional::variable({1});
    OpAlgoContext ctx("ReduceMean");

    y.add_child(x);
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
