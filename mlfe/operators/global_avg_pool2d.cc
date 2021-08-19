#include "mlfe/operators/global_avg_pool2d.h"
#include "mlfe/operators/utils.h"

namespace mlfe{
namespace operators{

Tensor global_average_pool2d(Tensor x)
{
    auto y = functional::create_variable({x.shape()[0], x.shape()[3]});
    auto y_weak = y.weak_copy();
    auto gm_x = [x, y_weak](Tensor &dy){
        global_average_pool2d_bwd_kernel::fn(x, y_weak, dy, x.grad());
    };
    call<global_average_pool2d_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators
} // namespace mlfe
