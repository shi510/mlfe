#ifndef __GRADIENT_CHECKER_H__
#define __GRADIENT_CHECKER_H__
#include "mlfe/core/tensor.h"
#include <numeric>
#include <functional>
#include <vector>

namespace mlfe{

// numerical = (func(theta + eps) - func(theta - eps)) / (2 * eps)
template <typename T>
Tensor numerical_gradient(const T eps, Tensor func, Tensor theta){
    // numerical gradient.
    Tensor result = functional::create_variable(theta.shape());
    Tensor temp = functional::create_variable(func.shape());
    
    for(int n = 0; n < theta.size(); ++n){
        auto val = theta.data<T>()[n];
        
        theta.mutable_data<T>()[n] = val + eps;
        func.eval();
        
        std::copy(func.cbegin<T>(),
                  func.cend<T>(),
                  temp.begin<T>()
                  );
        
        theta.mutable_data<T>()[n] = val - eps;
        func.eval();
        
        for(int k = 0; k < func.size(); ++k){
            auto tmp_ptr = temp.mutable_data<T>();
            auto func_ptr = func.data<T>();
            tmp_ptr[k] = (tmp_ptr[k] - func_ptr[k]) / (2 * eps);
        }
        
        T num_grad_nth = std::accumulate(temp.cbegin<T>(),
                                         temp.cend<T>(),
                                         T(0)
                                         );
        // restore original value
        theta.mutable_data<T>()[n] = val;
        result.mutable_data<T>()[n] = num_grad_nth;
    }
    return result;
}

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
T l2_norm(const mlfe::Tensor & x)
{
    T square_sum = T(0);
    for(int n = 0; n < x.size(); ++n){
        square_sum += x.data<T>()[n] * x.data<T>()[n];
    }
    return std::sqrt(square_sum);
}

template <typename T>
T l2_norm(const std::vector<T> & x)
{
    T square_sum = T(0);
    for(int n = 0; n < x.size(); ++n){
        square_sum += x[n] * x[n];
    }
    return std::sqrt(square_sum);
}

template <typename T>
std::vector<T> operator-(const mlfe::Tensor & a, const std::vector<T> & b)
{
    std::vector<T> c(b.size());
    for(int n = 0; n < b.size(); ++n)
    {
        c[n] = a.data<T>()[n] - b[n];
    }
    return c;
}

template <typename T>
T calculate_gradient_diff(const Tensor & numerical_grad, const std::vector<T> & analytical_grad)
{
    auto a_norm = l2_norm(analytical_grad);
    auto n_norm = l2_norm<T>(numerical_grad);
    auto norm = l2_norm(numerical_grad - analytical_grad);
    return norm / (a_norm + n_norm);
}

} // end namespace mlfe
#endif // end #ifndef __GRADIENT_CHECKER_H__
