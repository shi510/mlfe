#include "mlfe/operators/dropout.h"

namespace mlfe{
namespace operators{

Tensor dropout(Tensor x, float drop_ratio, bool is_training)
{
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor &dy){
        dropout_bwd_kernel::fn(x, dy, x.grad_v2(), drop_ratio);
    };
    call<dropout_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x),
        drop_ratio,
        is_training
    );
    return y;
}

} // namespace operators
} // namespace mlfe
