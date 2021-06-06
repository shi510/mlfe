#include "mlfe/operators_v2/sigmoid.h"

namespace mlfe{
namespace operators_v2{

Tensor sigmoid(Tensor x)
{
    auto y = functional::create_variable(x.shape());
    auto y_weak = y.weak_copy();
    auto gm_x = [x, y_weak](Tensor &dy){
        sigmoid_bwd_kernel::fn(y_weak, dy, x.grad_v2());
    };
    call<sigmoid_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators_v2
} // namespace mlfe