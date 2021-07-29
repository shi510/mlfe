#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators{

using relu_fwd_fn_t = std::function<void (Tensor, Tensor)>;
DECLARE_OP_KERNEL(relu_fwd, relu_fwd_fn_t);

using relu_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
DECLARE_OP_KERNEL(relu_bwd, relu_bwd_fn_t);

Tensor relu(Tensor x);

} // namespace operators
} // namespace mlfe
