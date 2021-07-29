#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void gradient_descent_momentum(const int size, T *w, const T *dw,
    T *w_momentum, const T lr, const T momentum, const T decay);

} // namespace cuda_kernel
} // namespace mlfe
