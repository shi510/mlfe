#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators{

using global_average_pool2d_fwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(global_average_pool2d_fwd, global_average_pool2d_fwd_fn_t);

using global_average_pool2d_bwd_fn_t =  std::function<void (Tensor, Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(global_average_pool2d_bwd, global_average_pool2d_bwd_fn_t);

Tensor global_average_pool2d(Tensor x);

} // namespace operators
} // namespace mlfe
