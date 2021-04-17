#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using reduce_mean_fwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(reduce_mean_fwd, reduce_mean_fwd_fn_t);

using reduce_mean_bwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(reduce_mean_bwd, reduce_mean_bwd_fn_t);

Tensor reduce_mean(Tensor input);

} // namespace operators_v2
} // namespace mlfe
