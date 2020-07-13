#pragma once
#include <cstdint>

namespace mlfe{
namespace util{

int32_t calc_conv2d_output(
    int input,
    int filter,
    int stride,
    int padding
    );

int32_t calc_conv2d_pad_size_for_same_output(
    int input,
    int filter,
    int stride
    );

} // end namespace util
} // end namespace mlfe
