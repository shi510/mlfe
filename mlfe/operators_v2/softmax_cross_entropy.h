#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using softmax_xent_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(softmax_xent_fwd, softmax_xent_fwd_fn_t);

using softmax_xent_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(softmax_xent_bwd, softmax_xent_bwd_fn_t);

Tensor softmax_cross_entropy(Tensor labels, Tensor logits);

} // namespace operators_v2
} // namespace mlfe
