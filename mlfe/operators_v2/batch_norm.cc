#include "mlfe/operators_v2/batch_norm.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators_v2{

Tensor batch_norm2d(
    Tensor x,
    Tensor scales,
    Tensor biases,
    Tensor rmean,
    Tensor rvar,
    bool trace_running_status)
{
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor &dy){
        batch_norm_bwd_kernel::fn(
            x, scales, dy, x.grad_v2(), scales.grad_v2(), biases.grad_v2());
    };
    call<batch_norm_fwd_kernel>(
        marker::I(x, scales, biases, rmean, rvar),
        marker::O(y)(gm_x),
        trace_running_status);
    return y;
}

} // namespace operators_v2
} // namespace mlfe
