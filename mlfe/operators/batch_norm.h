#pragma once
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace functional{

Tensor batch_normalize(Tensor x);

Tensor batch_normalize(Tensor x, Tensor scales, Tensor biases);

Tensor batch_normalize(Tensor x, Tensor scales, Tensor biases,
    Tensor mean, Tensor var, float eps=1e-5);

} // end namespace functional
} // end namespace mlfe

