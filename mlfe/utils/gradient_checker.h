#ifndef __GRADIENT_CHECKER_H__
#define __GRADIENT_CHECKER_H__
#include "mlfe/core/tensor.h"
#include <numeric>

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

} // end namespace mlfe
#endif // end #ifndef __GRADIENT_CHECKER_H__
