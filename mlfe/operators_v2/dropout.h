#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using dropout_fwd_fn_t = std::function<void (Tensor, Tensor, float, bool)>;
DECLARE_OP_KERNEL(dropout_fwd, dropout_fwd_fn_t);
using dropout_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, float)>;
DECLARE_OP_KERNEL(dropout_bwd, dropout_bwd_fn_t);

Tensor dropout(Tensor x, float drop_ratio, bool is_training);

} // namespace operators_v2
} // namespace mlfe
