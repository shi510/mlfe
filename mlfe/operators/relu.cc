#include "mlfe/operators/relu.h"

namespace mlfe{
namespace operators{

Tensor relu(Tensor x)
{
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor &dy){
        relu_bwd_kernel::fn(x, dy, x.grad());
    };
    call<relu_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators
} // namespace mlfe
