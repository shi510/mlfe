#pragma once
#include <cctype>
#include <cstddef>

namespace mlfe{
namespace cuda_kernel{

template <typename T>
void broadcast(
    const T *x, T *y,
    int Nx, int Cx, int Hx, int Wx,
    int Ny, int Cy, int Hy, int Wy);

template <typename T>
void broadcast_gradient(
    const T *dy, T *dx,
    int Ny, int Cy, int Hy, int Wy,
    int Nx, int Cx, int Hx, int Wx);

} // namespace cuda_kernel
} // namespace mlfe
