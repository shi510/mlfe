#include "mlfe/operators_v2/batch_norm1d.h"
#include "mlfe/core/op_kernel.h"
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace {


template <typename T>
struct batch_norm1d_nhwc_op
{
    static void run(
        Tensor x,
        Tensor scales,
        Tensor biases,
        Tensor rmean,
        Tensor rvar,
        Tensor y,
        bool track_running_status)
    {
        std::cout<<"No implementaion of batch_norm1d on CPU op."<<std::endl;
    }
};

template <class T>
struct batch_norm1d_nhwc_grad_op
{
    static void run(
        Tensor x,
        Tensor scales,
        Tensor dy,
        Tensor dx,
        Tensor dscales,
        Tensor dbiases)
    {
        std::cout<<"No implementaion of batch_norm1d gradient on CPU op."<<std::endl;
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    batch_norm1d_fwd,
    batch_norm_fwd_fn_t,
    batch_norm1d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    batch_norm1d_bwd,
    batch_norm_bwd_fn_t,
    batch_norm1d_nhwc_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
