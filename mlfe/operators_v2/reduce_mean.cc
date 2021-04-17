#include "mlfe/operators_v2/reduce_mean.h"

namespace mlfe{
namespace operators_v2{

Tensor reduce_mean(Tensor x)
{
    auto y = functional::create_variable({1});
    auto gm_x = [=](Tensor dy){
        reduce_mean_bwd_kernel::fn(dy, x.grad_v2());
    };
    call<reduce_mean_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators_v2
} // namespace mlfe
