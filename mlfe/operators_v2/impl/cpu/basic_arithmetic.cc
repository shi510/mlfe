#include "mlfe/operators_v2/basic_arithmetic.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace operators_v2{
namespace {

#define DEFINE_ELEMENTWISE_OP(name, expr)                  \
template <typename T>                                      \
void name##_fwd_impl(Tensor a, Tensor b, Tensor y){        \
    auto a_ptr = a.device_data<T>();                       \
    auto b_ptr = b.device_data<T>();                       \
    auto y_ptr = y.mutable_device_data<T>();               \
    int size = y.size();                                   \
    for(int n = 0; n < size; ++n){                         \
        y_ptr[n] = a_ptr[n] expr b_ptr[n];                 \
    }                                                      \
}

DEFINE_ELEMENTWISE_OP(eltwise_add, +)
DEFINE_ELEMENTWISE_OP(eltwise_sub, -)
DEFINE_ELEMENTWISE_OP(eltwise_mul, *)
DEFINE_ELEMENTWISE_OP(eltwise_div, /)

template <typename T>
void eltwise_add_left_bwd_impl(Tensor dy, Tensor da){
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = dy_ptr[n];
    }
}

template <typename T>
void eltwise_add_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        db_ptr[n] = dy_ptr[n];
    }
}

template <typename T>
void eltwise_sub_left_bwd_impl(Tensor dy, Tensor da){
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = dy_ptr[n];
    }
}

template <typename T>
void eltwise_sub_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        db_ptr[n] = -dy_ptr[n];
    }
}

template <typename T>
void eltwise_mul_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    auto b_ptr = b.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = b_ptr[n] * dy_ptr[n];
    }
}

template <typename T>
void eltwise_mul_right_bwd_impl(Tensor a, Tensor dy, Tensor db){
    auto a_ptr = a.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        db_ptr[n] = a_ptr[n] * dy_ptr[n];
    }
}

template <typename T>
void eltwise_div_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    auto b_ptr = b.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = dy_ptr[n] / b_ptr[n];
    }
}

template <typename T>
void eltwise_div_right_bwd_impl(Tensor b, Tensor y, Tensor dy, Tensor db){
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        db_ptr[n] = -dy_ptr[n] * y_ptr[n] / b_ptr[n];
    }
}

#define DEFINE_SCALAR_OP(name, expr)                       \
template <typename T>                                      \
void name##_fwd_impl(Tensor a, Tensor b, Tensor y){        \
    auto a_ptr = a.device_data<T>();                       \
    auto scalar = b.device_data<T>()[0];                   \
    auto y_ptr = y.mutable_device_data<T>();               \
    int size = y.size();                                   \
    for(int n = 0; n < size; ++n){                         \
        y_ptr[n] = a_ptr[n] expr scalar;                   \
    }                                                      \
}

DEFINE_SCALAR_OP(scalar_add, +)
DEFINE_SCALAR_OP(scalar_sub, -)
DEFINE_SCALAR_OP(scalar_mul, *)
DEFINE_SCALAR_OP(scalar_div, /)

template <typename T>
void scalar_add_left_bwd_impl(Tensor dy, Tensor da){
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = dy_ptr[n];
    }
}

template <typename T>
void scalar_add_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    T sum = T(0);
    for(int n = 0; n < size; ++n){
        sum += dy_ptr[n];
    }
    db_ptr[0] = sum;
}

template <typename T>
void scalar_sub_left_bwd_impl(Tensor dy, Tensor da){
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = dy_ptr[n];
    }
}

template <typename T>
void scalar_sub_right_bwd_impl(Tensor dy, Tensor db){
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    T sum = T(0);
    for(int n = 0; n < size; ++n){
        sum += -dy_ptr[n];
    }
    db_ptr[0] = sum;
}

template <typename T>
void scalar_mul_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    auto scalar_b = b.device_data<T>()[0];
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = scalar_b * dy_ptr[n];
    }
}

template <typename T>
void scalar_mul_right_bwd_impl(Tensor a, Tensor dy, Tensor db){
    auto a_ptr = a.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    T sum = T(0);
    for(int n = 0; n < size; ++n){
        sum += a_ptr[n] * dy_ptr[n];
    }
    db_ptr[0] = sum;
}

template <typename T>
void scalar_div_left_bwd_impl(Tensor b, Tensor dy, Tensor da){
    auto b_ptr = b.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto da_ptr = da.mutable_device_data<T>();
    auto size = dy.size();
    for(int n = 0; n < size; ++n){
        da_ptr[n] = b_ptr[n] * dy_ptr[n];
    }
}

template <typename T>
void scalar_div_right_bwd_impl(Tensor b, Tensor y, Tensor dy, Tensor db){
    auto scalar_b = b.device_data<T>()[0];
    auto y_ptr = y.device_data<T>();
    auto dy_ptr = dy.device_data<T>();
    auto db_ptr = db.mutable_device_data<T>();
    auto size = dy.size();
    T sum = T(0);
    for(int n = 0; n < size; ++n){
        sum += -dy_ptr[n] * y_ptr[n] / scalar_b;
    }
    db_ptr[0] = sum;
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

namespace{

template <typename T>
void set_zeros_fwd_impl(Tensor x){
    auto x_ptr = x.mutable_device_data<T>();
    for(int i = 0; i < x.size(); ++i){ x_ptr[i] = T(0); }
}

template <typename T>
void set_ones_fwd_impl(Tensor x){
    auto x_ptr = x.mutable_device_data<T>();
    for(int i = 0; i < x.size(); ++i){ x_ptr[i] = T(1); }
}

} // namespace anonymous

REGIST_OP_KERNEL(set_zeros_fwd, set_x_fwd_fn_t, set_zeros_fwd_impl<float>);
REGIST_OP_KERNEL(set_ones_fwd, set_x_fwd_fn_t, set_ones_fwd_impl<float>);

} // namespace operators
} // namespace mlfe
