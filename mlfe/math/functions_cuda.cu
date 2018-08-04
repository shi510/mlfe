#include "functions_cuda.h"
#include "functions.h"
#include <cub\block\block_reduce.cuh>
#include "../device_context/cuda_context.h"
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

template <class DataType>
__global__ void ReLUKernel(const int size, const DataType *x, DataType *y) {
    CUDA_1D_KERNEL_LOOP(i, size) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

template <class DataType>
__global__ void ReLUGradientKernel(const int size, const DataType *x, const DataType *dy, DataType *dx) {
    CUDA_1D_KERNEL_LOOP(i, size) {
        dx[i] = x[i] > 0 ? dy[i] : 0;
    }
}

template <>
void ReluFunction<float, CUDAContext>(const int size, const float *x, float *y) {
    ReLUKernel<float> << <CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS >> >(size, x, y);
}

template <>
void ReluGradientFunction<float, CUDAContext>(const int size, const float *x, const float *dy, float *dx) {
    ReLUGradientKernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x, dy, dx);
}


template <class DataType>
__global__ void SigmoidKernel(const int size, const DataType *x, DataType *y) {
    CUDA_1D_KERNEL_LOOP(i, size) {
        y[i] = 1.f / (1.f + exp(-x[i]));
    }
}

template <class DataType>
__global__  void SigmoidGradientKernel(const int size, const DataType *y, const DataType *dy, DataType *dx) {
    CUDA_1D_KERNEL_LOOP(i, size) {
        dx[i] = dy[i] * y[i] * (1. - y[i]);
    }
}

template <>
void SigmoidFunction<float, CUDAContext>(const int size, const float *x, float *y) {
    SigmoidKernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x, y);
}

template <>
void SigmoidGradientFunction<float, CUDAContext>(const int size, const float *y, const float *dy, float *dx) {
    SigmoidGradientKernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, y, dy, dx);
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

#define DEFINE_BINARY_OP_KERNEL(OpName, Expr) \
template <typename T>\
__global__ void OpName##_binary_op(const int size, const T *a, const T *b, T *c) {\
    CUDA_1D_KERNEL_LOOP(n, size){\
        c[n] = a[n] Expr b[n]; \
    }\
}\
template <typename T>\
__global__ void OpName##_val_binary_op(const int size, const T val, const T *a, T *c) {\
    CUDA_1D_KERNEL_LOOP(n, size){\
        c[n] = val Expr a[n]; \
    }\
}

DEFINE_BINARY_OP_KERNEL(Add, +)
DEFINE_BINARY_OP_KERNEL(Sub, -)
DEFINE_BINARY_OP_KERNEL(Mul, *)
DEFINE_BINARY_OP_KERNEL(Div, /)

#define DEFINE_CUDA_BINARY_OP(OpName) \
template <>\
void OpName##Cuda<float>(const int size, const float *a, const float *b, float *c){\
    OpName##_binary_op<float><<<\
    CUDA_CONTEXT_GET_BLOCKS(size), \
    CUDA_CONTEXT_NUM_THREADS>>>(size, a, b, c);\
}\
template <>\
void OpName##ValCuda<float>(const int size, const float val, const float *a, float *c) {\
    OpName##_val_binary_op<float><<<\
    CUDA_CONTEXT_GET_BLOCKS(size), \
    CUDA_CONTEXT_NUM_THREADS>>>(size, val, a, c);\
}\
template <>\
void OpName##Cuda<double>(const int size, const double *a, const double *b, double *c) {\
    OpName##_binary_op<double><<<\
    CUDA_CONTEXT_GET_BLOCKS(size), \
    CUDA_CONTEXT_NUM_THREADS>>>(size, a, b, c);\
}\
template <>\
void OpName##ValCuda<double>(const int size, const double val, const double *a, double *c) {\
    OpName##_val_binary_op<double><<<\
    CUDA_CONTEXT_GET_BLOCKS(size), \
    CUDA_CONTEXT_NUM_THREADS>>>(size, val, a, c);\
}

DEFINE_CUDA_BINARY_OP(Add)
DEFINE_CUDA_BINARY_OP(Sub)
DEFINE_CUDA_BINARY_OP(Mul)
DEFINE_CUDA_BINARY_OP(Div)

} // end namespace math
} // end namespace mlfe
