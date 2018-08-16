#ifndef __CONVOLUTION_OP_HPP__
#define __CONVOLUTION_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

Tensor Conv(Tensor x,
            Tensor w,
            type::int32::T filters,
            std::vector<type::int32::T> filters_hw,
            std::vector<type::int32::T> strides,
            std::vector<type::int32::T> pads
            );

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __CONVOLUTION_OP_HPP__
