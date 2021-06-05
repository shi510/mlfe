#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using sigmoid_fwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(sigmoid_fwd, sigmoid_fwd_fn_t);

using sigmoid_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(sigmoid_bwd, sigmoid_bwd_fn_t);

Tensor sigmoid(Tensor x);

} // namespace operators_v2
} // namespace mlfe
