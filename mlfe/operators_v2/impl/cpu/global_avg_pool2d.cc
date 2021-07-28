#include "mlfe/operators_v2/global_avg_pool2d.h"
#include "mlfe/core/op_kernel.h"
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
struct global_average_pool2d_nhwc_op
{
    static void run(Tensor x, Tensor y)
    {
        std::cout<<"No implementaion of global_average_pool2d on CPU op."<<std::endl;
    }
};

template <class T>
struct global_average_pool2d_nhwc_grad_op
{
    static void run(
        Tensor x,
        Tensor y,
        Tensor dy,
        Tensor dx
        )
    {
        std::cout<<"No implementaion of batch_norm1d gradient on CPU op."<<std::endl;
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    global_average_pool2d_fwd,
    global_average_pool2d_fwd_fn_t,
    global_average_pool2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    global_average_pool2d_bwd,
    global_average_pool2d_bwd_fn_t,
    global_average_pool2d_nhwc_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
