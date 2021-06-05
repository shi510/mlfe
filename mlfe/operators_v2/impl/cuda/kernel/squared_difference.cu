#include "mlfe/operators_v2/impl/cuda/kernel/squared_difference.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{
namespace cuda_kernel{

template <typename T> __global__
void squared_diff_kernel(const size_t N, const T *a_ptr, const T *b_ptr, T *y_ptr){
    CUDA_1D_KERNEL_LOOP(i, N){
        T diff = a_ptr[i] - b_ptr[i];
        y_ptr[i] = diff * diff;
    }
}

template <>
void squared_difference(const size_t size, const float *a_ptr, const float *b_ptr, float *y_ptr)
{
    squared_diff_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, a_ptr, b_ptr, y_ptr);
}

template <typename T> __global__
void squared_diff_bwd_kernel(const size_t N, const T scale, const T *a_ptr, const T *b_ptr, const T *dy_ptr, T *grad){
    const T val = 2;
    CUDA_1D_KERNEL_LOOP(i, N){
        grad[i] +=  scale * val * (a_ptr[i] - b_ptr[i]) * dy_ptr[i];
    }
}

template <>
void squared_difference_left_bwd(const size_t size, const float *a_ptr, const float *b_ptr, const float *dy_ptr, float *da_ptr)
{
    squared_diff_bwd_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, 1, a_ptr, b_ptr, dy_ptr, da_ptr);
}

template <>
void squared_difference_right_bwd(const size_t size, const float *a_ptr, const float *b_ptr, const float *dy_ptr, float *db_ptr)
{
    squared_diff_bwd_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, -1, a_ptr, b_ptr, dy_ptr, db_ptr);
}

} // namespace cuda_kernel
} // namespace mlfe