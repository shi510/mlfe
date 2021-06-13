#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <vector>
#include <cctype>

namespace mlfe{
namespace operators_v2{

using batch_norm_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool)>;
DECLARE_OP_KERNEL(batch_norm2d_fwd, batch_norm_fwd_fn_t);

using batch_norm_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(batch_norm2d_bwd, batch_norm_bwd_fn_t);

Tensor batch_norm2d(
    Tensor x,
    Tensor scales,
    Tensor biases,
    Tensor rmean,
    Tensor rvar,
    bool trace_running_status=true);

} // namespace operators_v2
} // namespace mlfe
