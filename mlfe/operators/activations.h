#ifndef __ACTIVATIONS_OP_HPP__
#define __ACTIVATIONS_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor ReLU(Tensor x);

Tensor Sigmoid(Tensor x);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __ACTIVATIONS_OP_HPP__
