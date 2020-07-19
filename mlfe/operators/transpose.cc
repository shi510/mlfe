#include "transpose.h"
#include "mlfe/core/gradient_helper.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/operators/initializer.h"

namespace mlfe{
namespace functional{

class TranspositionGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using Ints = std::vector<type::int32::T>;
        VecTensor in_grads;
        auto y_ctx = y.get_context();
        auto x = y_ctx.get_input(0);
        auto perm = y_ctx.get_input(1);
        Tensor dx;
        OpAlgoContext ctx("TranspositionGradient");
        ctx.add_input(x);
        ctx.add_input(perm);
        ctx.add_input(dy);
        ctx.add_output(dx);
        ctx.set_attrs(y_ctx.get_attrs());
        dx.set_context(ctx);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        auto dperm = functional::constant(1, perm.shape());
        perm.set_backprop_node(dperm.get_node());
        perm.set_gradient(dperm);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Transposition, TranspositionGradient)

Tensor transpose(Tensor x, const std::vector<int> perm)
{
    Tensor y;
    Tensor perm_tn = create_variable({(int)perm.size()});
    for(int n = 0; n < perm.size(); ++n){
        perm_tn.mutable_data<int>()[n] = perm[n];
    }
    OpAlgoContext ctx("Transposition");
    ctx.add_input(x);
    ctx.add_input(perm_tn);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
