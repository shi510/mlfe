#include "mlfe/operators/squared_difference.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators{
namespace {

template <typename T>
void squared_diff_fwd_impl(Tensor a, Tensor b, Tensor y)
{
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    auto size = y.size();
    for(int i = 0; i < size; ++i){
        T diff = a_ptr[i] - b_ptr[i];
        y_ptr[i] = diff * diff;
    }
}

template <typename T>
void squared_diff_left_bwd_impl(Tensor a, Tensor b, Tensor dy, Tensor da)
{
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = da.size();
    for(int i = 0; i < size; ++i){
        da_ptr[i] +=  T(2) * (a_ptr[i] - b_ptr[i]) * dy_ptr[i];
    }
}

template <typename T>
void squared_diff_right_bwd_impl(Tensor a, Tensor b, Tensor dy, Tensor db)
{
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = db.size();
    for(int i = 0; i < size; ++i){
        db_ptr[i] +=  T(-2) * (a_ptr[i] - b_ptr[i]) * dy_ptr[i];
    }
}

} // namespace anonymous

REGIST_OP_KERNEL(squared_diff_fwd, squared_diff_fwd_fn_t, squared_diff_fwd_impl<float>);
REGIST_OP_KERNEL(squared_diff_left_bwd, squared_diff_bwd_fn_t, squared_diff_left_bwd_impl<float>);
REGIST_OP_KERNEL(squared_diff_right_bwd, squared_diff_bwd_fn_t, squared_diff_right_bwd_impl<float>);

} // namespace operators
} // namespace mlfe
