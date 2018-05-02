#include "functions_cuda.hpp"
#include "functions.hpp"
#include <cub\block\block_reduce.cuh>
#include "../device_context/cuda_context.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
namespace mlfe { namespace math {

template <typename T>
__global__ void random_uniform_shift_kernel(
    const unsigned int size,
    T *numbers, const T a, const T b)
{
    const T scale = b - a;
    CUDA_1D_KERNEL_LOOP(n, size) {
        numbers[n] = numbers[n] * scale + a;
    }
}

template <>
void UniformCurand<float>(
    curandGenerator_t *gen,
    const unsigned int size,
    float *numbers, const float a, const float b)
{
    curandGenerateUniform(*gen, numbers, size);
    random_uniform_shift_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, numbers, a, b);
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


template <typename T>
__global__ void divide_by_val_kernel(const int val, T *arg) {
    arg[0] = arg[0] / static_cast<T>(val);
}

template <typename T>
__global__ void top_k_correct_count_kernel(const int batch, const int classes, const int top_k, const T *prob, const T *label, T *accuracy) {
    typedef cub::BlockReduce<int, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int correct = 0;
    for (int b = blockIdx.x; b < batch; b += gridDim.x) {
        const int gt = static_cast<int>(label[b]);
        const T gt_prob = prob[b * classes + gt];
        int rank = 0;
        for (int n = threadIdx.x; n < classes; n += blockDim.x) {
            const T prob_ = prob[b * classes + n];
            if (prob_ > gt_prob) {
                ++rank;
            }
        }
        rank = BlockReduce(temp_storage).Sum(rank);
        if (rank < top_k) {
            ++correct;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(accuracy, static_cast<T>(correct));
    }
}

template <>
void AccuracyCuda<float>(const int batch, const int classes, const int top_k, const float *prob, const float *label, float *accuracy) {
    top_k_correct_count_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(batch * classes),
        CUDA_CONTEXT_NUM_THREADS>>>(
            batch, classes, top_k, prob, label, accuracy
            );
    divide_by_val_kernel<float><<<1, 1>>>(batch, accuracy);
}

} // end namespace math
} // end namespace mlfe
