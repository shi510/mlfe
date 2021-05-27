#include "mlfe/operators_v2/basic_arithmetic.h"
#include "mlfe/operators_v2/broadcast.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/transform.h"
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators_v2{

Tensor elementwise_add(Tensor a, Tensor b)
{
    auto y = functional::create_variable(a.shape());
    auto gm_left = [=](Tensor &dy){
        eltwise_add_left_bwd_kernel::fn(dy, a.grad_v2());
    };
    auto gm_right = [=](Tensor &dy){
        eltwise_add_right_bwd_kernel::fn(dy, b.grad_v2());
    };
    call<eltwise_add_fwd_kernel>(
        marker::I(a, b),
        marker::O(y)(gm_left, gm_right));
    return y;
}

Tensor elementwise_sub(Tensor a, Tensor b)
{
    auto y = functional::create_variable(a.shape());
    auto gm_left = [=](Tensor &dy){
        eltwise_sub_left_bwd_kernel::fn(dy, a.grad_v2());
    };
    auto gm_right = [=](Tensor &dy){
        eltwise_sub_right_bwd_kernel::fn(dy, b.grad_v2());
    };
    call<eltwise_sub_fwd_kernel>(
        marker::I(a, b),
        marker::O(y)(gm_left, gm_right));
    return y;
}

Tensor elementwise_mul(Tensor a, Tensor b)
{
    auto y = functional::create_variable(a.shape());
    auto gm_left = [=](Tensor &dy){
        eltwise_mul_left_bwd_kernel::fn(b, dy, a.grad_v2());
    };
    auto gm_right = [=](Tensor &dy){
        eltwise_mul_right_bwd_kernel::fn(a, dy, b.grad_v2());
    };
    call<eltwise_mul_fwd_kernel>(
        marker::I(a, b),
        marker::O(y)(gm_left, gm_right));
    return y;
}

Tensor elementwise_div(Tensor a, Tensor b)
{
    auto y = functional::create_variable(a.shape());
    auto gm_left = [=](Tensor &dy){
        eltwise_div_left_bwd_kernel::fn(b, dy, a.grad_v2());
    };
    auto gm_right = [=](Tensor &dy){
        eltwise_div_right_bwd_kernel::fn(b, y, dy, b.grad_v2());
    };
    call<eltwise_div_fwd_kernel>(
        marker::I(a, b),
        marker::O(y)(gm_left, gm_right));
    return y;
}

Tensor scalar_add(Tensor a, Tensor scalar)
{
    auto y = functional::create_variable(a.shape());
    auto gm_a = [=](Tensor &dy){
        scalar_add_left_bwd_kernel::fn(dy, a.grad_v2());
    };
    auto gm_scalar = [=](Tensor &dy){
        scalar_add_right_bwd_kernel::fn(dy, scalar.grad_v2());
    };
    call<scalar_add_fwd_kernel>(
        marker::I(a, scalar),
        marker::O(y)(gm_a, gm_scalar));
    return y;
}

Tensor scalar_sub(Tensor a, Tensor scalar)
{
    auto y = functional::create_variable(a.shape());
    auto gm_a = [=](Tensor &dy){
        scalar_sub_left_bwd_kernel::fn(dy, a.grad_v2());
    };
    auto gm_scalar = [=](Tensor &dy){
        scalar_sub_right_bwd_kernel::fn(dy, scalar.grad_v2());
    };
    call<scalar_sub_fwd_kernel>(
        marker::I(a, scalar),
        marker::O(y)(gm_a, gm_scalar));
    return y;
}

Tensor scalar_mul(Tensor a, Tensor scalar)
{
    auto y = functional::create_variable(a.shape());
    auto gm_a = [=](Tensor &dy){
        scalar_mul_left_bwd_kernel::fn(scalar, dy, a.grad_v2());
    };
    auto gm_scalar = [=](Tensor &dy){
        scalar_mul_right_bwd_kernel::fn(a, dy, scalar.grad_v2());
    };
    call<scalar_mul_fwd_kernel>(
        marker::I(a, scalar),
        marker::O(y)(gm_a, gm_scalar));
    return y;
}

Tensor scalar_div(Tensor a, Tensor scalar)
{
    auto y = functional::create_variable(a.shape());
    auto gm_a = [=](Tensor &dy){
        scalar_div_left_bwd_kernel::fn(scalar, dy, a.grad_v2());
    };
    auto gm_scalar = [=](Tensor &dy){
        scalar_div_right_bwd_kernel::fn(scalar, y, dy, scalar.grad_v2());
    };
    call<scalar_div_fwd_kernel>(
        marker::I(a, scalar),
        marker::O(y)(gm_a, gm_scalar));
    return y;
}

template <typename T>
bool is_same(const std::vector<T> & a, const std::vector<T> & b){
    if(a.size() != b.size()){ return false; }
    for(int i = 0; i < a.size(); ++i){ if(a[i] != b[i]) { return false; } }
    return true;
}

#define DEFINE_ARITHMETIC_OP(op_name)                                        \
Tensor op_name(Tensor a, Tensor b){                                          \
    Tensor y;                                                                \
    if(is_same(a.shape(), b.shape())) { y = elementwise_##op_name(a, b); }   \
    else if(a.dims() != 0 && b.dims() == 0) { y = scalar_##op_name(a, b); }  \
    else if(a.dims() == 0 && b.dims() != 0) { y = scalar_##op_name(b, a); }  \
    else {                                                                   \
        auto to_shape = math::check_broadcasting(&a.shape(), &b.shape());    \
        if(to_shape.size() != 0 && is_same(to_shape, a.shape())){            \
            y = elementwise_##op_name(a, broadcast(b, to_shape));            \
        }                                                                    \
        else if(to_shape.size() != 0 && is_same(to_shape, b.shape())){       \
            y = elementwise_##op_name(broadcast(a, to_shape), b);            \
        }                                                                    \
        else{                                                                \
            std::stringstream ss;                                            \
            ss << "Can not broadcast ";                                      \
            for(auto n : a.shape()){ ss << n <<" ";}                         \
            ss << "<-> ";                                                    \
            for(auto n : b.shape()){ ss << n <<" ";}                         \
            std::runtime_error(ss.str());                                    \
        }                                                                    \
    }                                                                        \
    return y;                                                                \
}

DEFINE_ARITHMETIC_OP(add)
DEFINE_ARITHMETIC_OP(sub)
DEFINE_ARITHMETIC_OP(mul)
DEFINE_ARITHMETIC_OP(div)

void set_zeros(Tensor x){ call<set_zeros_fwd_kernel>(marker::I(x)); }
void set_ones(Tensor x){ call<set_ones_fwd_kernel>(marker::I(x)); }

} // namespace operators_v2
} // namespace mlfe
