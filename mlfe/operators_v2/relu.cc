#include "mlfe/operators_v2/relu.h"

namespace mlfe{
namespace operators_v2{

Tensor relu(Tensor x)
{
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor dy){
        relu_bwd_kernel::fn(x, dy, x.grad_v2());
    };
    call<relu_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators_v2
} // namespace mlfe
