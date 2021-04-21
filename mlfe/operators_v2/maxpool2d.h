#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{

using maxpool2d_fwd_fn_t = 
    std::function<void (Tensor, Tensor, std::vector<int32_t>, std::vector<int32_t>)>;
DECLARE_OP_KERNEL(maxpool2d_fwd, maxpool2d_fwd_fn_t);

using maxpool2d_bwd_fn_t = 
    std::function<void (Tensor, Tensor, Tensor, std::vector<int32_t>, std::vector<int32_t>)>;
DECLARE_OP_KERNEL(maxpool2d_bwd, maxpool2d_bwd_fn_t);

Tensor maxpool2d(Tensor x, std::vector<int32_t>, std::vector<int32_t>);

} // namespace operators_v2
} // namespace mlfe
