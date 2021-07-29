#include "mlfe/operators/batch_norm2d.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators{

Tensor batch_norm2d(
    Tensor x,
    Tensor scales,
    Tensor biases,
    Tensor rmean,
    Tensor rvar,
    bool trace_running_status)
{
    if(x.shape().size() != 4){
        throw std::runtime_error("batch_norm2d: input shape must be 4d.");
    }
    auto y = functional::create_variable(x.shape());
    auto gm_x = [=](Tensor &dy){
        batch_norm2d_bwd_kernel::fn(
            x, scales, dy, x.grad_v2(), scales.grad_v2(), biases.grad_v2());
    };
    call<batch_norm2d_fwd_kernel>(
        marker::I(x, scales, biases, rmean, rvar),
        marker::O(y)(gm_x),
        trace_running_status);
    return y;
}

} // namespace operators
} // namespace mlfe
