#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using sigmoid_xent_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(sigmoid_xent_fwd, sigmoid_xent_fwd_fn_t);

using sigmoid_xent_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(sigmoid_xent_bwd, sigmoid_xent_bwd_fn_t);

Tensor sigmoid_cross_entropy(Tensor labels, Tensor logits);

} // namespace operators_v2
} // namespace mlfe
