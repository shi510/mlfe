#include "mlfe/operators_v2/softmax_cross_entropy.h"
#include <tuple>

namespace mlfe{
namespace operators_v2{

Tensor softmax_cross_entropy(Tensor labels, Tensor logits)
{
    auto y = functional::create_variable({logits.shape()[0]});
    auto gm_x = [=](Tensor &dy){
        softmax_xent_bwd_kernel::fn(labels, logits, dy, logits.grad_v2());
    };
    call<softmax_xent_fwd_kernel>(
        marker::I(labels, logits),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators_v2
} // namespace mlfe
