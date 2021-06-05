#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void maxpool2d_nhwc(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const T* x_ptr,
    T* y_ptr);

template <typename T>
void maxpool2d_grad_nhwc(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const T* x_ptr,
    const T* y_ptr,
    const T* dy_ptr,
    T* dx_ptr);

} // namespace cuda_kernel
} // namespace mlfe
