#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <vector>
#include <cctype>

namespace mlfe{
namespace operators_v2{

using squared_diff_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(squared_diff_fwd, squared_diff_fwd_fn_t);

using squared_diff_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(squared_diff_left_bwd, squared_diff_bwd_fn_t);
DECLARE_OP_KERNEL(squared_diff_right_bwd, squared_diff_bwd_fn_t);

Tensor squared_difference(Tensor a, Tensor b);

} // namespace operators_v2
} // namespace mlfe
