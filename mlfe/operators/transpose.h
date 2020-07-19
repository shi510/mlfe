#pragma once
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace functional{

Tensor transpose(Tensor x, const std::vector<int> perm);

} // end namespace functional
} // end namespace mlfe

