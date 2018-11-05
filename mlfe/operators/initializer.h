#ifndef __INITIALIZER_OP_HPP__
#define __INITIALIZER_OP_HPP__
#include "../core/tensor.h"
#include <vector>

namespace mlfe{
namespace functional{

Tensor constant(type::float64::T val, std::vector<int> shape);

Tensor normal(type::float64::T std, std::vector<int> shape);

Tensor truncated_normal(type::float64::T std, std::vector<int> shape);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __INITIALIZER_OP_HPP__
