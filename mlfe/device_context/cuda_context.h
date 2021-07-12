#pragma once
#include <algorithm>
#include <functional>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include "context.h"

constexpr int CUDA_CONTEXT_NUM_THREADS = 512;
constexpr int CUDA_CONTEXT_MAXIMUM_NUM_BLOCKS = 2048;

#define CUDA_1D_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;      \
       i < (n);                                            \
       i += blockDim.x * gridDim.x)

inline int CUDA_CONTEXT_GET_BLOCKS(const int N) {
    return std::min<int>((N + CUDA_CONTEXT_NUM_THREADS - 1) / CUDA_CONTEXT_NUM_THREADS,
        CUDA_CONTEXT_MAXIMUM_NUM_BLOCKS);
}

namespace mlfe {

class CUDAContext final : public Context {
public:
    CUDAContext();

    ~CUDAContext() override;

    cublasHandle_t GetHandler() const;

private:
    static int static_shared_counter;
    static cublasHandle_t handler;
public:
    static curandGenerator_t rng;
};/* class CUDAContext */

struct cuda_context_v2 final : public Context {
    static std::shared_ptr<cuda_context_v2> create();

    ~cuda_context_v2() override;

    cublasHandle_t get_cublas_handle() const;

    cudnnHandle_t get_cudnn_handle() const;

    curandGenerator_t get_curand_generator() const;

private:
    cuda_context_v2();
    static int static_shared_counter;
    static cublasHandle_t cublas_handle;
    static cudnnHandle_t cudnn_handle;
    static curandGenerator_t curand_rng;
}; // struct cuda_context_v2

} // namespace mlfe
