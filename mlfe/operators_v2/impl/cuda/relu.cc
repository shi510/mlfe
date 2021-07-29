#include "mlfe/operators/relu.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace operators{
namespace {

template <typename T>
void relu_fwd_impl(Tensor x, Tensor y)
{
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    math::relu<T, CUDAContext>(x.size(), x_ptr, y_ptr);
}

template <typename T>
void relu_bwd_impl(Tensor x, Tensor dy, Tensor dx)
{
    auto x_ptr = x.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    auto size = x.size();
    math::relu_gradient<T, CUDAContext>(size, x_ptr, dy_ptr, dx_ptr);
}

} // namespace anonymous

REGIST_OP_KERNEL(relu_fwd, relu_fwd_fn_t, relu_fwd_impl<float>);
REGIST_OP_KERNEL(relu_bwd, relu_bwd_fn_t, relu_bwd_impl<float>);

} // namespace operators
} // namespace mlfe
