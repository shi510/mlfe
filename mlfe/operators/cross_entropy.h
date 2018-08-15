#ifndef __CROSS_ENTROPY_HPP__
#define __CROSS_ENTROPY_HPP__

#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor SigmoidCrossEntropy(Tensor x, Tensor y);

Tensor SoftmaxCrossEntropy(Tensor x, Tensor y);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __CROSS_ENTROPY_HPP__
