#include "mlfe/operators/maxpool2d.h"
#include "mlfe/operators/impl/cuda/kernel/maxpool2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include <cfloat>

namespace mlfe{
namespace operators{
namespace {

template <typename T>
struct maxpool2d_nhwc_op
{
    static void run(
                    Tensor x,
                    Tensor y,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        cuda_kernel::maxpool2d_nhwc<T>(
            x.shape()[0],
            x.shape()[3],
            x.shape()[1],
            x.shape()[2],
            y.shape()[1],
            y.shape()[2],
            psize[0],
            strides[0],
            x.device_data<T>(),
            y.mutable_device_data<T>());
    }
};

template <class T>
struct maxpool2d_nhwc_grad_op
{
    static void run(
                    Tensor x,
                    Tensor y,
                    Tensor dy,
                    Tensor dx,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        cuda_kernel::maxpool2d_grad_nhwc<T>(
            x.shape()[0],
            x.shape()[3],
            x.shape()[1],
            x.shape()[2],
            y.shape()[1],
            y.shape()[2],
            psize[0],
            strides[0],
            x.device_data<T>(),
            y.device_data<T>(),
            dy.device_data<T>(),
            dx.mutable_device_data<T>());
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    maxpool2d_fwd,
    maxpool2d_fwd_fn_t,
    maxpool2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    maxpool2d_bwd,
    maxpool2d_bwd_fn_t,
    maxpool2d_nhwc_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
