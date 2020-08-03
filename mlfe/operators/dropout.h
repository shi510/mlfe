#pragma once
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace functional{

Tensor dropout(Tensor x, Tensor keep_prob);

} // end namespace functional
} // end namespace mlfe
