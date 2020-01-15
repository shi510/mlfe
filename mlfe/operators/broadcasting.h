#ifndef __BROADCASTING_OP_HPP__
#define __BROADCASTING_OP_HPP__
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace functional{

Tensor broadcast(Tensor x,
                 std::vector<type::int32::T> shape
                );

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __BROADCASTING_OP_HPP__
