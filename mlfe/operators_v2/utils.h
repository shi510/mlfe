#pragma once
#include <cstdint>

namespace mlfe{
namespace operators{
namespace utils{

int32_t calc_conv_output(int input, int filter, int stride, int padding);
int32_t calc_conv_same_output_padding_size(int input, int filter, int stride);

} // namespace utils
} // namespace operators
} // namespace mlfe
