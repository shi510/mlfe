#ifndef __POOL_OP_HPP__
#define __POOL_OP_HPP__
#include "mlfe/core/tensor.h"
#include <vector>

namespace mlfe{
namespace functional{

Tensor pool_max(Tensor x,
                std::vector<int> kernel, 
                std::vector<int> stride, 
                std::vector<int> padding
               );

Tensor global_average_pool(Tensor x);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __POOL_OP_HPP__
