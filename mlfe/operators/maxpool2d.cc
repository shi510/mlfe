#include "mlfe/operators/maxpool2d.h"
#include "mlfe/operators/utils.h"

namespace mlfe{
namespace operators{

Tensor maxpool2d(Tensor x, std::vector<int32_t> psize, std::vector<int32_t> strides)
{
    int out_h = utils::calc_conv_output(x.shape()[1], psize[0], strides[0], 0);
    int out_w = utils::calc_conv_output(x.shape()[2], psize[1], strides[1], 0);
    auto y = functional::create_variable(
        {x.shape()[0], out_h, out_w, x.shape()[3]});
    // TODO :
    //  - New API Design for circular dependency.
    //  Some operators need self tensor to calculate its gradients.
    //  But when it holds self tensor, it does not free the own memory because of circular dependency of shared_ptr.
    // Do not copy directly, when it needs self tensor in gradient marker.
    // Use its weak copy to prevent circular dependency.
    auto y_weak = y.weak_copy();
    auto gm_x = [x, psize, strides, y_weak](Tensor &dy){
        maxpool2d_bwd_kernel::fn(x, y_weak, dy, x.grad(), psize, strides);
    };
    call<maxpool2d_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x),
        psize, strides);
    return y;
}

} // namespace operators
} // namespace mlfe
