#include "mlfe/operators_v2/basic_arithmetic.h"
#include "mlfe/operators_v2/impl/cudnn/kernel/basic_arithmetic.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cuda_context.h"


namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
void eltwise_add_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    math::AddCuda<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void eltwise_sub_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    math::SubCuda<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void eltwise_mul_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    math::MulCuda<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void eltwise_div_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    math::DivCuda<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void eltwise_add_left_bwd_impl(Tensor dy, Tensor da){
    mlfe::copy(dy.get_memory(), da.get_memory());
}

template <typename T>
void eltwise_add_right_bwd_impl(Tensor dy, Tensor db){
    mlfe::copy(dy.get_memory(), db.get_memory());
}

template <typename T>
void eltwise_sub_left_bwd_impl(Tensor dy, Tensor da){
    mlfe::copy(dy.get_memory(), da.get_memory());
}

template <typename T>
void eltwise_sub_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    math::scal<T, CUDAContext>(size, T(-1), dy_ptr, db_ptr);
}

template <typename T>
void eltwise_mul_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    eltwise_mul_fwd_impl<T>(b, dy, da);
}

template <typename T>
void eltwise_mul_right_bwd_impl(Tensor a, Tensor dy, Tensor db){
    eltwise_mul_fwd_impl<T>(a, dy, db);
}

template <typename T>
void eltwise_div_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    eltwise_div_fwd_impl<T>(dy, b, da);
}

template <typename T>
void eltwise_div_right_bwd_impl(Tensor b, Tensor y, Tensor dy, Tensor db){
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    cuda_kernel::eltwise_div_right_bwd(size, dy_ptr, b_ptr, y_ptr, db_ptr);
}

template <typename T>
void scalar_add_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    cuda_kernel::scalar_add_fwd<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void scalar_sub_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    cuda_kernel::scalar_sub_fwd<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void scalar_mul_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    cuda_kernel::scalar_mul_fwd<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void scalar_div_fwd_impl(Tensor a, Tensor b, Tensor y){
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();
    int size = y.size();
    cuda_kernel::scalar_div_fwd<T>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T>
void scalar_add_left_bwd_impl(Tensor dy, Tensor da){
    mlfe::copy(dy.get_memory(), da.get_memory());
}

template <typename T>
void scalar_add_right_bwd_impl(Tensor dy, Tensor db){
    math::sum<T, CUDAContext>(
        dy.size(),
        dy.device_data<T>(),
        db.mutable_device_data<T>());
}

template <typename T>
void scalar_sub_left_bwd_impl(Tensor dy, Tensor da){
    mlfe::copy(dy.get_memory(), da.get_memory());
}

template <typename T>
void scalar_sub_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    cuda_kernel::scalar_sub_right_bwd<T>(size, dy_ptr, db_ptr);
}

template <typename T>
void scalar_mul_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    auto scalar_b = b.data<T>()[0];
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    math::scal<T, CUDAContext>(size, scalar_b, dy_ptr, da_ptr);
}

template <typename T>
void scalar_mul_right_bwd_impl(Tensor a, Tensor dy, Tensor db){
    auto a_ptr = a.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    cuda_kernel::scalar_mul_right_bwd<T>(size, a_ptr, dy_ptr, db_ptr);
}

template <typename T>
void scalar_div_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    eltwise_mul_fwd_impl<T>(b, dy, da);
}

template <typename T>
void scalar_div_right_bwd_impl(Tensor b, Tensor y, Tensor dy, Tensor db){
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    cuda_kernel::scalar_div_right_bwd<T>(size, b_ptr, y_ptr, dy_ptr, db_ptr);
}

} // namespace anonymous

REGIST_OP_KERNEL(eltwise_add_fwd, arithmetic_fwd_fn_t, eltwise_add_fwd_impl<float>);
REGIST_OP_KERNEL(eltwise_add_left_bwd, add_left_bwd_fn_t, eltwise_add_left_bwd_impl<float>);
REGIST_OP_KERNEL(eltwise_add_right_bwd, add_right_bwd_fn_t, eltwise_add_right_bwd_impl<float>);

REGIST_OP_KERNEL(eltwise_sub_fwd, arithmetic_fwd_fn_t, eltwise_sub_fwd_impl<float>);
REGIST_OP_KERNEL(eltwise_sub_left_bwd, sub_left_bwd_fn_t, eltwise_sub_left_bwd_impl<float>);
REGIST_OP_KERNEL(eltwise_sub_right_bwd, sub_right_bwd_fn_t, eltwise_sub_right_bwd_impl<float>);

REGIST_OP_KERNEL(eltwise_mul_fwd, arithmetic_fwd_fn_t, eltwise_mul_fwd_impl<float>);
REGIST_OP_KERNEL(eltwise_mul_left_bwd, mul_left_bwd_fn_t, eltwise_mul_left_bwd_impl<float>);
REGIST_OP_KERNEL(eltwise_mul_right_bwd, mul_right_bwd_fn_t, eltwise_mul_right_bwd_impl<float>);

REGIST_OP_KERNEL(eltwise_div_fwd, arithmetic_fwd_fn_t, eltwise_div_fwd_impl<float>);
REGIST_OP_KERNEL(eltwise_div_left_bwd, div_left_bwd_fn_t, eltwise_div_left_bwd_impl<float>);
REGIST_OP_KERNEL(eltwise_div_right_bwd, div_right_bwd_fn_t, eltwise_div_right_bwd_impl<float>);


REGIST_OP_KERNEL(scalar_add_fwd, arithmetic_fwd_fn_t, scalar_add_fwd_impl<float>);
REGIST_OP_KERNEL(scalar_add_left_bwd, add_left_bwd_fn_t, scalar_add_left_bwd_impl<float>);
REGIST_OP_KERNEL(scalar_add_right_bwd, add_right_bwd_fn_t, scalar_add_right_bwd_impl<float>);

REGIST_OP_KERNEL(scalar_sub_fwd, arithmetic_fwd_fn_t, scalar_sub_fwd_impl<float>);
REGIST_OP_KERNEL(scalar_sub_left_bwd, sub_left_bwd_fn_t, scalar_sub_left_bwd_impl<float>);
REGIST_OP_KERNEL(scalar_sub_right_bwd, sub_right_bwd_fn_t, scalar_sub_right_bwd_impl<float>);

REGIST_OP_KERNEL(scalar_mul_fwd, arithmetic_fwd_fn_t, scalar_mul_fwd_impl<float>);
REGIST_OP_KERNEL(scalar_mul_left_bwd, mul_left_bwd_fn_t, scalar_mul_left_bwd_impl<float>);
REGIST_OP_KERNEL(scalar_mul_right_bwd, mul_right_bwd_fn_t, scalar_mul_right_bwd_impl<float>);

REGIST_OP_KERNEL(scalar_div_fwd, arithmetic_fwd_fn_t, scalar_div_fwd_impl<float>);
REGIST_OP_KERNEL(scalar_div_left_bwd, div_left_bwd_fn_t, scalar_div_left_bwd_impl<float>);
REGIST_OP_KERNEL(scalar_div_right_bwd, div_right_bwd_fn_t, scalar_div_right_bwd_impl<float>);


} // namespace operators
} // namespace mlfe
