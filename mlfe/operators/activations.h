#ifndef __ACTIVATIONS_OP_HPP__
#define __ACTIVATIONS_OP_HPP__
#include "mlfe/core/tensor.h"

namespace mlfe{ namespace functional{

Tensor relu(Tensor x);

Tensor sigmoid(Tensor x);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __ACTIVATIONS_OP_HPP__
