#ifndef __CONVOLUTION_OP_HPP__
#define __CONVOLUTION_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{
namespace functional{

Tensor conv2d(Tensor x,
              Tensor w,
              std::vector<type::int32::T> strides,
              std::vector<type::int32::T> pads
              );

Tensor depthwise_conv2d(Tensor x,
    Tensor w,
    std::vector<type::int32::T> strides,
    std::vector<type::int32::T> pads
    );

Tensor conv2d(Tensor x, Tensor w, std::vector<int32_t> strides, bool same_out);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __CONVOLUTION_OP_HPP__
