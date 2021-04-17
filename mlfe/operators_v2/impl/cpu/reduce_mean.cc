#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/operators_v2/reduce_mean.h"

namespace mlfe{
namespace operators_v2{
namespace impl_cpu{

template <typename T>
void reduce_mean_fwd_impl(Tensor x, Tensor y){
    auto size = x.size();
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    T sum = 0;
    for(int n = 0; n < size; ++n){
        sum += x_ptr[n];
    }
    y_ptr[0] = sum / T(size);
}

REGIST_OP_KERNEL(
    reduce_mean_fwd,
    reduce_mean_fwd_fn_t,
    impl_cpu::reduce_mean_fwd_impl<float>
    );

template <typename T>
void reduce_mean_bwd_impl(Tensor dy, Tensor dx){
    auto size = dx.size();
    auto scale = T(1) / T(size);
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    T dy_val = dy_ptr[0];
    
    for(int n = 0; n < size; ++n){
        dx_ptr[n] = dy_val * scale;
    }
}

REGIST_OP_KERNEL(
    reduce_mean_bwd,
    reduce_mean_bwd_fn_t,
    impl_cpu::reduce_mean_bwd_impl<float>
    );

} // namespace impl_cpu
} // namespace op_kernels_cpu
} // namespace mlfe
