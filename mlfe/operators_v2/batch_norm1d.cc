#include "mlfe/operators_v2/batch_norm1d.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators_v2{

Tensor batch_norm1d(
    Tensor x,
    Tensor scales,
    Tensor biases,
    Tensor rmean,
    Tensor rvar,
    bool trace_running_status)
{
    if(x.shape().size() != 2){
        throw std::runtime_error("batch_norm1d: input shape must be 2d.");
    }
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor &dy){
        batch_norm1d_bwd_kernel::fn(
            x, scales, dy, x.grad_v2(), scales.grad_v2(), biases.grad_v2());
    };
    call<batch_norm1d_fwd_kernel>(
        marker::I(x, scales, biases, rmean, rvar),
        marker::O(y)(gm_x),
        trace_running_status);
    return y;
}

} // namespace operators_v2
} // namespace mlfe
