#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void col2im_nhwc(
    const T* data_col,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const int pad,
    T* data_im);

} // namespace cuda_kernel
} // namespace mlfe
