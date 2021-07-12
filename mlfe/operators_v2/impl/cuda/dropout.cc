#include "mlfe/operators_v2/dropout.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cuda_context.h"
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
void dropout_fwd_impl(Tensor x, Tensor y, T drop_ratio, bool is_training)
{
    auto size = x.size();
    auto mask = create_memory(size * sizeof(T));
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    auto mask_ptr = mask->mutable_device_data<T>();
    auto bernouli_fn = math::bernoulli_distribution<T, CUDAContext>;
    x.get_node().add_attr("mask", mask);

    T keep_ratio = T(1) - drop_ratio;
    T keep_ratio_inv = T(1) / keep_ratio;
    if(is_training && drop_ratio != 0){
        bernouli_fn(size, drop_ratio, mask_ptr);
        math::scal<T, CUDAContext>(size, keep_ratio_inv, mask_ptr, y_ptr);
        math::elementwise_mul<T, CUDAContext>(size, x_ptr, y_ptr, y_ptr);
    }
    else{
        copy(x.get_memory(), y.get_memory());
    }
}

template <typename T>
void dropout_bwd_impl(Tensor x, Tensor dy, Tensor dx, T drop_ratio)
{
    auto mask = *x.get_node().get_attr("mask").data<memory_ptr>();
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    auto mask_ptr = mask->device_data<T>();
    auto size = x.size();
    T keep_ratio = T(1) - drop_ratio;
    T keep_ratio_inv = T(1) / keep_ratio;
    math::scal<T, CUDAContext>(size, keep_ratio_inv, mask_ptr, dx_ptr);
    math::elementwise_mul<T, CUDAContext>(size, dy_ptr, dx_ptr, dx_ptr);
}

} // namespace anonymous

REGIST_OP_KERNEL(dropout_fwd, dropout_fwd_fn_t, dropout_fwd_impl<float>);
REGIST_OP_KERNEL(dropout_bwd, dropout_bwd_fn_t, dropout_bwd_impl<float>);

} // namespace operators
} // namespace mlfe
