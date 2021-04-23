#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <vector>
#include <cctype>

namespace mlfe{
namespace operators_v2{

using broadcast_fwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(broadcast_fwd, broadcast_fwd_fn_t);

using broadcast_bwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(broadcast_bwd, broadcast_bwd_fn_t);

Tensor broadcast(Tensor x, std::vector<int32_t> shape);

} // namespace operators_v2
} // namespace mlfe
