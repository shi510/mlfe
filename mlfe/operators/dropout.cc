#include "dropout.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"

namespace mlfe{ namespace functional{

class DropoutGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        auto x = ctx_y.get_input(0);
        auto dx = functional::create_variable(x.shape());
        OpAlgoContext ctx_dx("DropoutGradient");
        ctx_dx.add_input(x);
        ctx_dx.add_input(dy);
        ctx_dx.add_output(dx);
        ctx_dx.add_attr({"mask", ctx_y.get_attr<Tensor>("mask")});
        ctx_dx.add_attr({"keep_prob", ctx_y.get_attr<Tensor>("keep_prob")});
        dx.set_context(ctx_dx);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Dropout, DropoutGradient)

Tensor dropout(Tensor x, Tensor keep_prob){
    Tensor y = create_variable(x.shape());
    Tensor dropout_mask = create_variable(x.shape());
    OpAlgoContext ctx("Dropout");
    ctx.add_input(x);
    ctx.add_output(y);
    ctx.add_attr({"mask", dropout_mask});
    ctx.add_attr({"keep_prob", keep_prob});
    y.set_context(ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
