#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void adam(const int size, T *w, const T *dw, T *m_hist, T *v_hist,
    const T lr, const T beta1, const T beta2, const T eps);

} // namespace cuda_kernel
} // namespace mlfe
