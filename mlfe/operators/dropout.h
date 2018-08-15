#ifndef __DROPOUT_OP_HPP__
#define __DROPOUT_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor Dropout(Tensor x, type::float64::T probability, bool is_training);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __DROPOUT_OP_HPP__
