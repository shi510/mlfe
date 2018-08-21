#ifndef __INITIALIZER_OP_HPP__
#define __INITIALIZER_OP_HPP__
#include "../core/tensor.h"
#include <vector>

namespace mlfe{ namespace functional{

Tensor Constant(type::float64::T val, std::vector<int> shape = {});

Tensor Normal(type::float64::T std, std::vector<int> shape = {});

Tensor TruncatedNormal(type::float64::T std, std::vector<int> shape = {});

Tensor Xavier(type::int32::T a, type::int32::T b, std::vector<int> shape = {});

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __INITIALIZER_OP_HPP__
