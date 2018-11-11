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
        y.reshape({ 1 }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(ReduceMean)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.reshape(x.shape(), type::float32());
    })
    .Finish();

class ReduceMeanGradient : public GradientHelper{
public:
    ReduceMeanGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("ReduceMeanGradient");
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor mean(Tensor x){
    Tensor y = create_variable({1});
    OpAlgoContext ctx("ReduceMean");

    y.add_child(x);
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
