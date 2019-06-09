#include "dropout.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

class DropoutGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto x = y.get_children()[0];
        auto dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("DropoutGradient");
        auto mask = y.get_context().get_attr<Tensor>("mask");
        auto prob = y.get_context().get_attr<Tensor>("prob");
        dx.add_child(x);
        dx.add_child(dy);
        ctx.add_attr({"mask", mask});
        ctx.add_attr({"prob", prob});
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Dropout, DropoutGradient)

Tensor dropout(Tensor x, Tensor prob){
    Tensor y = create_variable(x.shape());
    Tensor dropout_mask = create_variable(x.shape());
    OpAlgoContext ctx("Dropout");
    y.add_child(x);
    ctx.add_attr({"mask", dropout_mask});
    ctx.add_attr({"prob", prob});
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
