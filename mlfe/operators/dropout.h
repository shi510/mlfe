#ifndef __DROPOUT_OP_HPP__
#define __DROPOUT_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{
namespace functional{

Tensor dropout(Tensor x, Tensor prob);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __DROPOUT_OP_HPP__
