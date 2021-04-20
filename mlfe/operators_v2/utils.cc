#include "mlfe/operators_v2/utils.h"
#include <algorithm>
#include <cmath>

namespace mlfe{
namespace operators_v2{
namespace utils{

int32_t calc_conv_output(int input, int filter, int stride, int padding)
{
    return (input - filter + 2 * padding) / stride + 1;
}

int32_t calc_conv_same_output_padding_size(int input, int filter, int stride)
{
    int out = std::floor(input / (float)stride);
    return std::max(out * stride + filter - input, 0) / 2;
}

} // namespace utils
} // namespace operators_v2
} // namespace mlfe
