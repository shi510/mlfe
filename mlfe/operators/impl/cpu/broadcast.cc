#include "mlfe/operators/broadcast.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/transform.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace operators{
namespace {

template <typename T>
void broadcast_fwd_impl(Tensor x, Tensor y){
    auto x_ptr = x.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    std::vector<int32_t> x_shape(4);
    std::vector<int32_t> y_shape(4);
    std::fill(x_shape.begin(), x_shape.end(), 1);
    std::fill(y_shape.begin(), y_shape.end(), 1);
    std::copy(x.shape().rbegin(), x.shape().rend(), x_shape.rbegin());
    std::copy(y.shape().rbegin(), y.shape().rend(), y_shape.rbegin());
    // broadcast
    math::broadcast<T, CPUContext>(x_ptr, y_ptr,
        x_shape[0], x_shape[1], x_shape[2], x_shape[3],
        y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
}

template <typename T>
void broadcast_bwd_impl(Tensor dy, Tensor dx){
    auto dy_ptr = dy.device_data<T>();
    auto dx_ptr = dx.mutable_device_data<T>();
    std::vector<int32_t> dy_shape(4);
    std::vector<int32_t> dx_shape(4);
    std::fill(dy_shape.begin(), dy_shape.end(), 1);
    std::fill(dx_shape.begin(), dx_shape.end(), 1);
    std::copy(dy.shape().rbegin(), dy.shape().rend(), dy_shape.rbegin());
    std::copy(dx.shape().rbegin(), dx.shape().rend(), dx_shape.rbegin());
    // zero
    math::set<T, CPUContext>(dx.size(), T(0), dx_ptr);
    math::broadcast_gradient<T, CPUContext>(dy_ptr, dx_ptr,
        dy_shape[0], dy_shape[1], dy_shape[2], dy_shape[3],
        dx_shape[0], dx_shape[1], dx_shape[2], dx_shape[3]);
}

} // namespace anonymous

REGIST_OP_KERNEL(broadcast_fwd, broadcast_fwd_fn_t, broadcast_fwd_impl<float>);
REGIST_OP_KERNEL(broadcast_bwd, broadcast_bwd_fn_t, broadcast_bwd_impl<float>);

} // namespace operators
} // namespace mlfe
