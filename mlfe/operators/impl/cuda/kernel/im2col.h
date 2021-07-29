#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void im2col_nhwc(
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int KH,
    const int KW,
    const int stride,
    const int padding,
    const T *im_ptr,
    T *col_ptr);

} // namespace cuda_kernel
} // namespace mlfe
