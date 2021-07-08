#ifndef __GRADIENT_CHECKER_H__
#define __GRADIENT_CHECKER_H__
#include "mlfe/core/tensor.h"
#include <numeric>
#include <functional>
#include <vector>

namespace mlfe{

// numerical = (func(theta + eps) - func(theta - eps)) / (2 * eps)
template <typename F, typename T>
Tensor numerical_gradient_v2(F func, Tensor theta, const T eps){
    Tensor result = functional::create_variable(theta.shape());

    for(int n = 0; n < theta.size(); ++n){
        auto val = theta.data<T>()[n];
        theta.mutable_data<T>()[n] = val + eps;
        Tensor j_p = func(theta);
        theta.mutable_data<T>()[n] = val - eps;
        Tensor j_n = func(theta);
        auto p_ptr = j_p.data<T>();
        auto n_ptr = j_n.data<T>();
        T num_grad_nth = T(0);
        for(int k = 0; k < j_p.size(); ++k){
            num_grad_nth += (p_ptr[k] - n_ptr[k]) / (2.f * eps);
        }
        // restore original value
        theta.mutable_data<T>()[n] = val;
        result.mutable_data<T>()[n] = num_grad_nth;
    }
    return result;
}

template <typename T>
T l2_norm(const Tensor & x)
{
    T square_sum = T(0);
    for(int n = 0; n < x.size(); ++n){
        square_sum += x.data<T>()[n] * x.data<T>()[n];
    }
    return std::sqrt(square_sum);
}

template <typename T>
T calculate_gradient_diff(const Tensor & numerical_grad, const Tensor & analytical_grad)
{
    auto a_norm = l2_norm<T>(analytical_grad);
    auto n_norm = l2_norm<T>(numerical_grad);
    auto norm = l2_norm<T>(numerical_grad - analytical_grad);
    return norm / (a_norm + n_norm);
}

} // end namespace mlfe
#endif // end #ifndef __GRADIENT_CHECKER_H__
