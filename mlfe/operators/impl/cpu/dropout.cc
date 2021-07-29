#include "mlfe/operators/dropout.h"
#include "mlfe/core/op_kernel.h"
#include <iostream>

namespace mlfe{
namespace operators{
namespace {

template <typename T>
void dropout_fwd_impl(Tensor x, Tensor y, T drop_ratio, bool is_training)
{
    std::cout<<"No implementaion of dropout on CPU op."<<std::endl;
}

template <typename T>
void dropout_bwd_impl(Tensor x, Tensor dy, Tensor dx, T drop_ratio)
{
    std::cout<<"No implementaion of dropout gradient on CPU op."<<std::endl;
}

} // namespace anonymous

REGIST_OP_KERNEL(dropout_fwd, dropout_fwd_fn_t, dropout_fwd_impl<float>);
REGIST_OP_KERNEL(dropout_bwd, dropout_bwd_fn_t, dropout_bwd_impl<float>);

} // namespace operators
} // namespace mlfe
