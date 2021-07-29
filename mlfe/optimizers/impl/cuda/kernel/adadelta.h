#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void adadelta(const int size, T *w, const T *dw, T *grad_hist,
    T *acc_hist, const T lr, const T momentum, const T eps);

} // namespace cuda_kernel
} // namespace mlfe
