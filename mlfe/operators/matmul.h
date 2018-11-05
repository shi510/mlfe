#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__
#include "../core/tensor.h"

namespace mlfe{
namespace functional{

Tensor matmul(Tensor a, 
              Tensor b, 
              bool trans_a = false, 
              bool trans_b = false
             );

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __MATMUL_HPP__
