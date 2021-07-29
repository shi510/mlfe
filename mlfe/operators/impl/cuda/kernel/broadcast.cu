#include "mlfe/operators/impl/cuda/kernel/broadcast.h"

namespace mlfe{
namespace cuda_kernel{


template <typename T>
__global__ void broadcast_kernel(
    const T *x, T *y,
    int Nx, int Cx, int Hx, int Wx,
    int Ny, int Cy, int Hy, int Wy)
{
    auto k = (blockIdx.y * blockDim.y) + threadIdx.y;
    auto l = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(k < Hy && l < Wy){
        for(int i = 0; i < Ny; ++i){
           for(int j = 0; j < Cy; ++j){
               int x_idx = (i % Nx) * Cx * Hx * Wx +
                           (j % Cx) * Hx * Wx +
                           (k % Hx) * Wx +
                           (l % Wx);
               int y_idx = i * Cy * Hy * Wy +
                           j * Hy * Wy +
                           k * Wy +
                           l;

               y[y_idx] = x[x_idx];
            }
        }
    }
}

template <>
void broadcast<float>(
    const float *x, float *y,
    int Nx, int Cx, int Hx, int Wx,
    int Ny, int Cy, int Hy, int Wy)
{
    dim3 n_threads = {16, 16};
    dim3 n_blocks = {32, 32};
    broadcast_kernel<float><<<n_blocks, n_threads>>>(
        x, y,
        Nx, Cx, Hx, Wx,
        Ny, Cy, Hy, Wy);
}

template <typename T>
__global__ void broadcast_gradient_kernel(
    const T *dy, T *dx,
    int Ny, int Cy, int Hy, int Wy,
    int Nx, int Cx, int Hx, int Wx)
{
    auto k = (blockIdx.y * blockDim.y) + threadIdx.y;
    auto l = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(k < Hy && l < Wy){
        for(int i = 0; i < Ny; ++i){
            for(int j = 0; j < Cy; ++j){
                int dx_idx = (i % Nx) * Cx * Hx * Wx +
                             (j % Cx) * Hx * Wx +
                             (k % Hx) * Wx +
                             (l % Wx);
                int dy_idx = i * Cy * Hy * Wy +
                             j * Hy * Wy +
                             k * Wy +
                             l;
                atomicAdd(dx + dx_idx, dy[dy_idx]);
            }
        }
    }
}

template <>
void broadcast_gradient<float>(
    const float *dy, float *dx,
    int Ny, int Cy, int Hy, int Wy,
    int Nx, int Cx, int Hx, int Wx)
{
    dim3 n_threads = {16, 16};
    dim3 n_blocks = {32, 32};
    broadcast_gradient_kernel<float><<<n_blocks, n_threads>>>(
        dy, dx,
        Ny, Cy, Hy, Wy,
        Nx, Cx, Hx, Wx);
}


} // namespace cuda_kernel
} // namespace mlfe
