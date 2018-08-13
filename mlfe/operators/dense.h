#ifndef __DENSE_HPP__
#define __DENSE_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor Dense(Tensor x, type::int32::T num_out, Tensor init_w, Tensor init_b);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __DENSE_HPP__
