#include "mlfe/operators_v2/impl/cudnn/kernel/basic_arithmetic.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{
namespace cuda_kernel{

template <class T> __global__
void negative_kernel(const size_t N, const T *x_ptr, T *y_ptr){
    CUDA_1D_KERNEL_LOOP(i, N){
        y_ptr[i] = -x_ptr[i];
    }
}

template <>
void negative<float>(const size_t N, const float *x_ptr, float *y_ptr){
    negative_kernel<<<CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, x_ptr, y_ptr);
}

template <class T> __global__
void scalar_add_fwd_kernel(const size_t N, const T *a, const T *b, T *c){
    T scalar = b[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        c[i] = a[i] + scalar;
    }
}

template <>
void scalar_add_fwd<float>(
    const size_t N,
    const float *a,
    const float *b,
    float *c)
{
    scalar_add_fwd_kernel<<<CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, b, c);
}

template <class T> __global__
void scalar_sub_fwd_kernel(const size_t N, const T *a, const T *b, T *c){
    T scalar = b[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        c[i] = a[i] - scalar;
    }
}

template <>
void scalar_sub_fwd<float>(
    const size_t N,
    const float *a,
    const float *b,
    float *c)
{
    scalar_sub_fwd_kernel<<<CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, b, c);
}

template <class T> __global__
void scalar_mul_fwd_kernel(const size_t N, const T *a, const T *b, T *c){
    T scalar = b[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        c[i] = a[i] * scalar;
    }
}

template <>
void scalar_mul_fwd<float>(
    const size_t N,
    const float *a,
    const float *b,
    float *c)
{
    scalar_mul_fwd_kernel<<<CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, b, c);
}

template <class T> __global__
void scalar_div_fwd_kernel(const size_t N, const T *a, const T *b, T *c){
    T scalar = b[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        c[i] = a[i] / scalar;
    }
}

template <>
void scalar_div_fwd<float>(
    const size_t N,
    const float *a,
    const float *b,
    float *c)
{
    scalar_div_fwd_kernel<<<CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, b, c);
}

template <typename T>
__global__ void eltwise_div_right_bwd_kernel(
    const size_t N,
    const T *dy,
    const T *b,
    const T *y,
    T *db)
{
    CUDA_1D_KERNEL_LOOP(i, N){
        db[i] = -dy[i] * y[i] / b[i];
    }
}

template <>
void eltwise_div_right_bwd(
    const size_t N,
    const float *dy,
    const float *b,
    const float *y,
    float *db)
{
    eltwise_div_right_bwd_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, dy, b, y, db);
}

template <int BLOCK_THREADS, typename T>
__global__ void scalar_sub_right_bwd_kernel(
    const size_t N,
    const T *dy,
    T *db)
{
    typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage smem_storage;

    T data;
    if(blockIdx.x * BLOCK_THREADS + threadIdx.x < N){
        data = -dy[blockIdx.x * BLOCK_THREADS + threadIdx.x];
    }

    T aggregate = BlockReduce(smem_storage).Sum(data);
    if(threadIdx.x == 0){
        atomicAdd(db, aggregate);
    }
}

template <>
void scalar_sub_right_bwd(
    const size_t N,
    const float *dy,
    float *db)
{
    scalar_sub_right_bwd_kernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, dy, db);
}

template <int BLOCK_THREADS, typename T>
__global__ void scalar_mul_right_bwd_kernel(
    const size_t N,
    const T *a,
    const T *dy,
    T *db)
{
    typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage smem_storage;

    T data;
    if(blockIdx.x * BLOCK_THREADS + threadIdx.x < N){
        data = a[blockIdx.x * BLOCK_THREADS + threadIdx.x] *
            dy[blockIdx.x * BLOCK_THREADS + threadIdx.x];
    }

    T aggregate = BlockReduce(smem_storage).Sum(data);
    if(threadIdx.x == 0){
        atomicAdd(db, aggregate);
    }
}

template<>
void scalar_mul_right_bwd<float>(
    const size_t N,
    const float *a,
    const float *dy,
    float *db)
{
    scalar_mul_right_bwd_kernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, dy, db);
}

template <int BLOCK_THREADS, typename T>
__global__ void scalar_div_right_bwd_kernel(
    const size_t N,
    const T *b,
    const T *y,
    const T *dy,
    T *db)
{
    typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage smem_storage;
    T scalar_b = b[0];
    T data;
    if(blockIdx.x * BLOCK_THREADS + threadIdx.x < N){
        data = -b[blockIdx.x * BLOCK_THREADS + threadIdx.x] *
            dy[blockIdx.x * BLOCK_THREADS + threadIdx.x] / scalar_b;
    }

    T aggregate = BlockReduce(smem_storage).Sum(data);
    if(threadIdx.x == 0){
        atomicAdd(db, aggregate);
    }
}

template <>
void scalar_div_right_bwd<float>(
    const size_t N,
    const float *b,
    const float *y,
    const float *dy,
    float*db)
{
    scalar_div_right_bwd_kernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, y, dy, db);
}

} // namespace cuda_kernel
} // namespace mlfe
