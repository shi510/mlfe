#include "mlfe/operators_v2/maxpool2d.h"
#include "mlfe/operators_v2/utils.h"

namespace mlfe{
namespace operators_v2{

Tensor maxpool2d(Tensor x, std::vector<int32_t> psize, std::vector<int32_t> strides)
{
    int out_h = utils::calc_conv_output(
        x.shape()[1], psize[0], strides[0], 0);
    int out_w = utils::calc_conv_output(
        x.shape()[2], psize[1], strides[1], 0);
    auto y =
        functional::create_variable({x.shape()[0], out_h, out_w, x.shape()[3]});
    auto gm_x = [=](Tensor dy){
        maxpool2d_bwd_kernel::fn(x, dy, x.grad_v2(), psize, strides);
    };
    call<maxpool2d_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x),
        psize, strides);
    return y;
}

} // namespace operators_v2
} // namespace mlfe
