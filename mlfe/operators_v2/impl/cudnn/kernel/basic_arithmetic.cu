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
__global__ void scaled_accum(
    const size_t N,
    const T scale,
    const T *from,
    T *to)
{
    CUDA_1D_KERNEL_LOOP(i, N){
        to[i] += scale * from[i];
    }
}

template <typename T>
__global__ void scaled_accum(
    const size_t N,
    const T *scale,
    const T *from,
    T *to)
{
    T val = scale[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        to[i] += val * from[i];
    }
}

template <>
void eltwise_add_left_bwd(const size_t N, const float *dy, float *da)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, da);
}

template <>
void eltwise_add_right_bwd(const size_t N, const float *dy, float *db)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, db);
}

template <>
void eltwise_sub_left_bwd(const size_t N, const float *dy, float *da)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, da);
}

template <>
void eltwise_sub_right_bwd(const size_t N, const float *dy, float *db)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, -1, dy, db);
}

template <typename T>
__global__ void mul_accum(
    const size_t N,
    const T *a,
    const T *b,
    T *c)
{
    CUDA_1D_KERNEL_LOOP(i, N){
        c[i] += a[i] * b[i];
    }
}

template <>
void eltwise_mul_left_bwd(const size_t N, const float *b, const float *dy, float *da)
{
    mul_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, dy, da);
}

template <>
void eltwise_mul_right_bwd(const size_t N, const float *a, const float *dy, float *db)
{
    mul_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, a, dy, db);
}

template <typename T>
__global__ void eltwise_div_left_bwd_kernel(
    const size_t N,
    const T *b,
    const T *dy,
    T *da)
{
    CUDA_1D_KERNEL_LOOP(i, N){
        da[i] += dy[i] / b[i];
    }
}

template <>
void eltwise_div_left_bwd(
    const size_t N,
    const float *b,
    const float *dy,
    float *da)
{
    eltwise_div_left_bwd_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, dy, da);
}

template <typename T>
__global__ void eltwise_div_right_bwd_kernel(
    const size_t N,
    const T *b,
    const T *y,
    const T *dy,
    T *db)
{
    CUDA_1D_KERNEL_LOOP(i, N){
        db[i] += -dy[i] * y[i] / b[i];
    }
}

template <>
void eltwise_div_right_bwd(
    const size_t N,
    const float *b,
    const float *y,
    const float *dy,
    float *db)
{
    eltwise_div_right_bwd_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, y, dy, db);
}

template <int BLOCK_THREADS, typename T>
__global__ void scaled_sum_kernel(
    const size_t N,
    const T scale,
    const T *dy,
    T *db)
{
    typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage smem_storage;
    size_t gid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    T data = 0;
    if(gid < N){
        data = scale * dy[gid];
    }

    T aggregate = BlockReduce(smem_storage).Sum(data);
    if(threadIdx.x == 0){
        atomicAdd(db, aggregate);
    }
}

template <>
void scalar_add_left_bwd(const size_t N, const float *dy, float *da)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, da);
}

template <>
void scalar_add_right_bwd(const size_t N, const float *dy, float *db)
{
    scaled_sum_kernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, db);
}

template <>
void scalar_sub_left_bwd(const size_t N, const float *dy, float *da)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, 1, dy, da);
}

template <>
void scalar_sub_right_bwd(
    const size_t N,
    const float *dy,
    float *db)
{
    scaled_sum_kernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, -1, dy, db);
}

template <>
void scalar_mul_left_bwd(const size_t N, const float *b, const float *dy, float *da)
{
    scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, dy, da);
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
    size_t gid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    T data = 0;
    if(gid < N){
        data = a[gid] * dy[gid];
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

template <typename T>
__global__ void inv_scaled_accum(
    const size_t N,
    const T *scale,
    const T *vec,
    T *out)
{
    T val = T(1) / scale[0];
    CUDA_1D_KERNEL_LOOP(i, N){
        out[i] += val * vec[i];
    }
}

template <>
void scalar_div_left_bwd(
    const size_t N,
    const float *b,
    const float *dy,
    float *da)
{
    inv_scaled_accum<float><<<
        CUDA_CONTEXT_GET_BLOCKS(N),
        CUDA_CONTEXT_NUM_THREADS>>>(N, b, dy, da);
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
    size_t gid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    T inv_b = T(1) / b[0];
    T data = 0;
    if(gid < N){
        data = -dy[gid] * y[gid] * inv_b;
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
