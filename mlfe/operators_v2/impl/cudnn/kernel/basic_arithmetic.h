#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

// fwd

template <typename T>
void negative(const size_t size, const T *x_ptr, T *y_ptr);

template <typename T>
void scalar_add_fwd(const size_t size, const T *a, const T *b, T *c);

template <typename T>
void scalar_sub_fwd(const size_t size, const T *a, const T *b, T *c);

template <typename T>
void scalar_mul_fwd(const size_t size, const T *a, const T *b, T *c);

template <typename T>
void scalar_div_fwd(const size_t size, const T *a, const T *b, T *c);

// bwd

template <typename T>
void eltwise_div_right_bwd(const size_t N, const T *dy, const T *b, const T *y, T *db);

template <typename T>
void scalar_sub_right_bwd(const size_t N, const T *dy, T *db);

template <typename T>
void scalar_mul_right_bwd(const size_t N, const T *a, const T *dy, T *db);

template <typename T>
void scalar_div_right_bwd(const size_t N, const T *b, const T *y, const T *dy, T *db);

} // namespace cuda_kernel
} // namespace mlfe
