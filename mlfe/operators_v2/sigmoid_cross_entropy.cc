#include "mlfe/operators_v2/sigmoid_cross_entropy.h"
#include <tuple>

namespace mlfe{
namespace operators_v2{

Tensor sigmoid_cross_entropy(Tensor labels, Tensor logits)
{
    auto y = functional::create_variable({logits.shape()[0]});
    auto gm_labels = [](Tensor &dy){};
    auto gm_logits = [=](Tensor &dy){
        sigmoid_xent_bwd_kernel::fn(labels, logits, dy, logits.grad_v2());
    };
    call<sigmoid_xent_fwd_kernel>(
        marker::I(labels, logits),
        marker::O(y)(gm_labels, gm_logits));
    return y;
}

} // namespace operators_v2
} // namespace mlfe
