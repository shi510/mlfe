#ifndef __RESHAPE_OP_HPP__
#define __RESHAPE_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor Reshape(Tensor x, std::vector<type::int32::T> shape);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __RESHAPE_OP_HPP__
