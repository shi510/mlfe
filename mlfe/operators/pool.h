#ifndef __POOL_OP_HPP__
#define __POOL_OP_HPP__
#include "../core/tensor.h"
#include <vector>

namespace mlfe{namespace functional{

Tensor MaxPool(Tensor x,
               std::vector<type::int32::T> filters_hw,
               std::vector<type::int32::T> strides,
               std::vector<type::int32::T> pads
               );

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __POOL_OP_HPP__
