#include "mlfe/operators/reduce_mean.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/basic_functions.h"

namespace mlfe{
namespace operators{
namespace {

template <typename T>
void reduce_mean_fwd_impl(Tensor x, Tensor y){
    auto size = x.size();
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    math::set<T, CUDAContext>(1, T(0), y_ptr);
    math::sum<T, CUDAContext>(size, x_ptr, y_ptr);
    math::scal<T, CUDAContext>(1, T(1) / T(size), y_ptr, y_ptr);
}

template <typename T>
void reduce_mean_bwd_impl(Tensor dy, Tensor dx){
    auto size = dx.size();
    auto scale = T(1) / T(size);
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    math::reduce_mean_gradient<T, CUDAContext>(size, scale, dy_ptr, dx_ptr);
}

} // namespace anonymous

REGIST_OP_KERNEL(
    reduce_mean_fwd,
    reduce_mean_fwd_fn_t,
    reduce_mean_fwd_impl<float>
    );

REGIST_OP_KERNEL(
    reduce_mean_bwd,
    reduce_mean_bwd_fn_t,
    reduce_mean_bwd_impl<float>
    );

} // namespace operators
} // namespace mlfe
