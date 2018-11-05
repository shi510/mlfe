#ifndef __CROSS_ENTROPY_HPP__
#define __CROSS_ENTROPY_HPP__

#include "../core/tensor.h"

namespace mlfe{
namespace functional{

Tensor softmax_cross_entropy(Tensor logit, Tensor label);

Tensor sigmoid_cross_entropy(Tensor logit, Tensor label);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __CROSS_ENTROPY_HPP__
