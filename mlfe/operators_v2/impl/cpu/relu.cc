#include "mlfe/operators_v2/relu.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace operators_v2{
namespace impl_cpu{

template <typename T>
void relu_fwd_impl(Tensor x, Tensor y){
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    math::relu<T, CPUContext>(x.size(), x_ptr, y_ptr);
}

REGIST_OP_KERNEL(relu_fwd, relu_fwd_fn_t, impl_cpu::relu_fwd_impl<float>);

template <typename T>
void relu_bwd_impl(Tensor x, Tensor dy, Tensor dx){
    auto x_ptr = x.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    auto size = x.size();
    math::relu_gradient<T, CPUContext>(size, x_ptr, dy_ptr, dx_ptr);
}

REGIST_OP_KERNEL(relu_bwd, relu_bwd_fn_t, impl_cpu::relu_bwd_impl<float>);

} // namespace impl_cpu
} // namespace op_kernels_cpu
} // namespace mlfe
