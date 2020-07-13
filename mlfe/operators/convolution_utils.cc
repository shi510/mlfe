#include "mlfe/operators/convolution_utils.h"
#include <algorithm>
#include <cmath>

namespace mlfe{
namespace util{

int32_t calc_conv2d_output(
    int input,
    int filter,
    int stride,
    int padding
    )
{
    return (input - filter + 2 * padding) / stride + 1;
}

int32_t calc_conv2d_pad_size_for_same_output(
    int input,
    int filter,
    int stride
    )
{
    int out = std::floor(input / (float)stride);
    return std::max(out * stride + filter - input, 0) / 2;
}

} // end namespace util
} // end namespace mlfe