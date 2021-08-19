#include "mlfe/operators/reduce_mean.h"

namespace mlfe{
namespace operators{

Tensor reduce_mean(Tensor x)
{
    auto y = functional::create_variable({1});
    auto gm_x = [=](Tensor &dy){
        reduce_mean_bwd_kernel::fn(dy, x.grad());
    };
    call<reduce_mean_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators
} // namespace mlfe
