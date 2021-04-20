#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <vector>

namespace mlfe{
namespace operators_v2{

using conv2d_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor, std::vector<int32_t>, std::vector<int32_t>)>;
DECLARE_OP_KERNEL(conv2d_fwd, conv2d_fwd_fn_t);

using conv2d_input_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, std::vector<int32_t>, std::vector<int32_t>)>;
DECLARE_OP_KERNEL(conv2d_input_bwd, conv2d_input_bwd_fn_t);

using conv2d_kernel_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, std::vector<int32_t>, std::vector<int32_t>)>;
DECLARE_OP_KERNEL(conv2d_kernel_bwd, conv2d_kernel_bwd_fn_t);

Tensor conv2d(
    Tensor x,
    Tensor kernel,
    std::vector<int32_t> strides,
    std::vector<int32_t> paddings);

Tensor conv2d(
    Tensor x,
    Tensor kernel,
    std::vector<int32_t> strides,
    bool same_out=false);

} // namespace operators_v2
} // namespace mlfe
