#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <vector>

namespace mlfe{
namespace operators_v2{

using arithmetic_fwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;


using add_left_bwd_fn_t = std::function<void (Tensor, Tensor)>;
using add_right_bwd_fn_t = std::function<void (Tensor, Tensor)>;

using sub_left_bwd_fn_t = std::function<void (Tensor, Tensor)>;
using sub_right_bwd_fn_t = std::function<void (Tensor, Tensor)>;

using mul_left_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
using mul_right_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;

using div_left_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor)>;
using div_right_bwd_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor)>;

DECLARE_OP_KERNEL(eltwise_add_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(eltwise_add_left_bwd, add_left_bwd_fn_t);
DECLARE_OP_KERNEL(eltwise_add_right_bwd, add_right_bwd_fn_t);

DECLARE_OP_KERNEL(eltwise_sub_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(eltwise_sub_left_bwd, sub_left_bwd_fn_t);
DECLARE_OP_KERNEL(eltwise_sub_right_bwd, sub_right_bwd_fn_t);

DECLARE_OP_KERNEL(eltwise_mul_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(eltwise_mul_left_bwd, mul_left_bwd_fn_t);
DECLARE_OP_KERNEL(eltwise_mul_right_bwd, mul_right_bwd_fn_t);

DECLARE_OP_KERNEL(eltwise_div_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(eltwise_div_left_bwd, div_left_bwd_fn_t);
DECLARE_OP_KERNEL(eltwise_div_right_bwd, div_right_bwd_fn_t);

DECLARE_OP_KERNEL(scalar_add_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(scalar_add_left_bwd, add_left_bwd_fn_t);
DECLARE_OP_KERNEL(scalar_add_right_bwd, add_right_bwd_fn_t);

DECLARE_OP_KERNEL(scalar_sub_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(scalar_sub_left_bwd, sub_left_bwd_fn_t);
DECLARE_OP_KERNEL(scalar_sub_right_bwd, sub_right_bwd_fn_t);

DECLARE_OP_KERNEL(scalar_mul_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(scalar_mul_left_bwd, mul_left_bwd_fn_t);
DECLARE_OP_KERNEL(scalar_mul_right_bwd, mul_right_bwd_fn_t);

DECLARE_OP_KERNEL(scalar_div_fwd, arithmetic_fwd_fn_t);
DECLARE_OP_KERNEL(scalar_div_left_bwd, div_left_bwd_fn_t);
DECLARE_OP_KERNEL(scalar_div_right_bwd, div_right_bwd_fn_t);

Tensor add(Tensor a, Tensor b);
Tensor sub(Tensor a, Tensor b);
Tensor mul(Tensor a, Tensor b);
Tensor div(Tensor a, Tensor b);


void add_inplace(Tensor a, Tensor b);
void sub_inplace(Tensor a, Tensor b);
void mul_inplace(Tensor a, Tensor b);
void div_inplace(Tensor a, Tensor b);

} // namespace operators_v2
} // namespace mlfe
