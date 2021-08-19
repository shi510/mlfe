#include "mlfe/operators/squared_difference.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators{

Tensor squared_difference(Tensor a, Tensor b)
{
    auto y = functional::create_variable(a.shape());
    auto gm_a = [a, b](Tensor &dy){
        squared_diff_left_bwd_kernel::fn(a, b, dy, a.grad());
    };
    auto gm_b = [a, b](Tensor &dy){
        squared_diff_right_bwd_kernel::fn(a, b, dy, b.grad());
    };
    call<squared_diff_fwd_kernel>(
        marker::I(a, b),
        marker::O(y)(gm_a, gm_b));
    return y;
}

} // namespace operators
} // namespace mlfe
