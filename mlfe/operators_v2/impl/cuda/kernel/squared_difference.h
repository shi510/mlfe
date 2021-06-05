#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void squared_difference(const size_t size, const T *a_ptr, const T *b_ptr, T *y_ptr);

template <typename T>
void squared_difference_left_bwd(const size_t size, const T *a_ptr, const T *b_ptr, const T *dy_ptr, T *da_ptr);

template <typename T>
void squared_difference_right_bwd(const size_t size, const T *a_ptr, const T *b_ptr, const T *dy_ptr, T *db_ptr);

} // namespace cuda_kernel
} // namespace mlfe
