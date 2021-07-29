#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators{

using matmul_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor, bool, bool)>;
DECLARE_OP_KERNEL(matmul_fwd, matmul_fwd_fn_t);

Tensor matmul(Tensor a, Tensor b, bool transpose_a=false, bool transpose_b=false);

} // namespace operators
} // namespace mlfe
