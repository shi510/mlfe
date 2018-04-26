#include "functions_cuda.hpp"
#include <cub\block\block_reduce.cuh>
#include "../device_context/cuda_context.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
namespace mlfe { namespace math {

__global__ void curand_init_kernel(const int size, unsigned int seed, curandState_t *states) {
    CUDA_1D_KERNEL_LOOP(n, size) {
        curand_init(seed, n, 0, &states[n]);
    }
}

template <typename T>
__global__ void curand_uniform_kernel(curandState_t *states, const int size, T *numbers, T a, T b) {
    CUDA_1D_KERNEL_LOOP(n, size) {
        numbers[n] = curand_uniform(&states[n]);
        numbers[n] = numbers[n] * (b - a) + a;
    }
}

void InitCurand(unsigned int seed, unsigned int n, curandState_t* states) {
    curand_init_kernel<<<CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS >>>(n, seed, states);
}

template <>
void UniformCurand<float>(curandState_t *states, unsigned int n, float *numbers, float a, float b) {
    curand_uniform_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS >>>(states, n, numbers, a, b);
}

template <typename T>
__global__ void one_hot_kernel(const int classes, const T *label, T *onehot) {
    int n = threadIdx.x;
    int label_val = static_cast<int>(label[n]);
    onehot[n * classes + label_val] = static_cast<T>(1);
}

template <>
void OneHotCuda<float>(const int batch, const int classes, const float *label, float *onehot) {
    one_hot_kernel<float><<<1, batch>>>(classes, label, onehot);
}


} // end namespace math
} // end namespace mlfe
