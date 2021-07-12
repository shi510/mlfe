#include "basic_functions.h"
#include <third_party/cub/cub/block/block_reduce.cuh>
#include "mlfe/math/blas.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace math{

template <class T> __global__
void negative_kernel(const int size,
                     const T *x_ptr,
                     T *y_ptr){
    CUDA_1D_KERNEL_LOOP(n, size){
        y_ptr[n] = -x_ptr[n];
    }
}

template <>
void negative<float, CUDAContext>(const int size,
                                  const float *x_ptr,
                                  float *y_ptr
                                 ){
    negative_kernel <<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, y_ptr);
}

template <class T> __global__
void squared_difference_kernel(const int size,
                               const T *x1_ptr,
                               const T *x2_ptr,
                               T *y_ptr){
    CUDA_1D_KERNEL_LOOP(n, size){
        y_ptr[n] = std::pow(x1_ptr[n] - x2_ptr[n], 2);
    }
}

template <>
void squared_difference<float, CUDAContext>(const int size,
                                            const float *x1_ptr,
                                            const float *x2_ptr,
                                            float *y_ptr){
    squared_difference_kernel<<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x1_ptr, x2_ptr, y_ptr);
}

template <>
void squared_difference<double, CUDAContext>(const int size,
                                             const double *x1_ptr,
                                             const double *x2_ptr,
                                             double *y_ptr){
    squared_difference_kernel<<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x1_ptr, x2_ptr, y_ptr);
}

template <class DataType> __global__
void rowwise_max_kernel(const int rows,
                        const int cols,
                        const DataType *data, DataType *out
                       )
{
    typedef cub::BlockReduce<float, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for(int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x){
        DataType maxval = static_cast<DataType>(-FLT_MAX);
        for(int colIndex = threadIdx.x; colIndex < cols; colIndex += blockDim.x){
            maxval = max(data[rowIndex * cols + colIndex], maxval);
        }
        maxval = BlockReduce(temp_storage).Reduce(maxval, cub::Max());
        if(threadIdx.x == 0){
            out[rowIndex] = maxval;
        }
        __syncthreads();
    }
}

template <> void
rowwise_max<float, CUDAContext>(const int m,
                                const int n,
                                const float *a_ptr,
                                float *b_ptr
                               )
{
    rowwise_max_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, a_ptr, b_ptr);
}

template <> void
rowwise_max<double, CUDAContext>(const int m,
                                 const int n,
                                 const double *a_ptr,
                                 double *b_ptr
                                )
{
    rowwise_max_kernel<double><<<
        CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, a_ptr, b_ptr);
}

template <class DT>
__global__ void rowwise_normalize_kernel(const int nthreads,
                                         const int D,
                                         const DT* scales,
                                         DT* out
                                        )
{
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int n = index / D;
        out[index] /= scales[n];
    }
}

template <> void
rowwise_normalize<float, CUDAContext>(const int m,
                                      const int n,
                                      const float *scaler_ptr,
                                      float *norm_dest
                                     )
{
    rowwise_normalize_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m * n, n, scaler_ptr, norm_dest);
}

template <> void
rowwise_normalize<double, CUDAContext>(const int m,
                                       const int n,
                                       const double *scaler_ptr,
                                       double *norm_dest
                                      )
{
    rowwise_normalize_kernel<double><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m * n, n, scaler_ptr, norm_dest);
}


template <class DataType> __global__
void exp_kernel(const int size,
                const DataType *x,
                DataType *y
               )
{
    CUDA_1D_KERNEL_LOOP(index, size){
        y[index] = std::exp(x[index]);
    }
}

template<>
void exp<float, CUDAContext>(const int size,
                             const float *x_ptr,
                             float *y_ptr
                            )
{
    exp_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, y_ptr);
}

template<>
void exp<double, CUDAContext>(const int size,
                              const double *x_ptr,
                              double *y_ptr
                             )
{
    exp_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, y_ptr);
}

template <class DataType> __global__
void axpy_kernel(const int size,
                 const DataType a,
                 const DataType *x,
                 DataType *y
                )
{
    CUDA_1D_KERNEL_LOOP(index, size){
        y[index] = a * x[index] + y[index];
    }
}

template<> void
axpy<float, CUDAContext>(int size,
                         const float alpha,
                         const float *x_ptr,
                         float *y_ptr
                        )
{
    axpy_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, alpha, x_ptr, y_ptr);
}

template<> void
axpy<double, CUDAContext>(int size,
                          const double alpha,
                          const double *x_ptr,
                          double *y_ptr
                         )
{
    axpy_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, alpha, x_ptr, y_ptr);
}

template <class DataType> __global__
void scale_kernel(const int size,
                  const DataType *x,
                  const DataType a,
                  DataType *y
                 )
{
    CUDA_1D_KERNEL_LOOP(index, size){
        y[index] = a * x[index];
    }
}

template <> void
scal<float, CUDAContext>(const int size,
                         const float alpha,
                         const float *x_ptr,
                         float *y_ptr
                        )
{
    scale_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, alpha, y_ptr);
}

template <> void
scal<double, CUDAContext>(const int size,
                          const double alpha,
                          const double *x_ptr,
                          double *y_ptr
                         )
{
    scale_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, alpha, y_ptr);
}

// This code refference from https://gist.github.com/yzhwang/5120437
template <int BLOCK_THREADS, typename DataType> __global__
void SumKernel(const int N,
               const DataType* X,
               DataType* Y
              )
{
    typedef cub::BlockReduce<DataType, BLOCK_THREADS> BlockReduce;

    __shared__ typename BlockReduce::TempStorage smem_storage;

    DataType data = 0;
    if(blockIdx.x * BLOCK_THREADS + threadIdx.x < N){
        data = X[blockIdx.x * BLOCK_THREADS + threadIdx.x];
    }

    DataType aggregate = BlockReduce(smem_storage).Sum(data);
    if(threadIdx.x == 0){
        atomicAdd(Y, aggregate);
    }
}

template <>
void sum<float, CUDAContext>(const int size,
                             const float *x_ptr,
                             float *y_ptr
                            )
{
    SumKernel<CUDA_CONTEXT_NUM_THREADS, float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS >>>(size, x_ptr, y_ptr);
}

template <class DataType> __global__
void set_kernel(const int size,
                const DataType val,
                DataType *x
               )
{
    CUDA_1D_KERNEL_LOOP(index, size){
        x[index] = val;
    }
}

template<>
void set<float, CUDAContext>(const int size,
                             const float val,
                             float *x_ptr
                            )
{
    set_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, val, x_ptr);
}

template<>
void set<double, CUDAContext>(const int size,
                              const double val,
                              double *x_ptr
                             )
{
    set_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, val, x_ptr);
}

template <class DataType> __global__
void elementsize_mul_kernel(const int size,
                            const DataType *a,
                            const DataType *b,
                            DataType *c
                           )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        c[n] = a[n] * b[n];
    }
}

template<>
void elementwise_mul<float, CUDAContext>(const int size,
                                         const float *a,
                                         const float *b,
                                         float *c
                                        )
{
    elementsize_mul_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, a, b, c);
}

template <typename T>
__global__ void clip_min_max_kernel(const int size,
                                    T *data,
                                    T min,
                                    T max
                                   )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        if(data[n] > max){
            data[n] = max;
        }
        else if(data[n] < min){
            data[n] = min;
        }
    }
}

template <>
void clip_min_max<float, CUDAContext>(const int size,
                                      float *data,
                                      float min,
                                      float max
                                     )
{
    clip_min_max_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, data, min, max);
}

template <typename T>
__global__ void shift_a_b_kernel(const unsigned int size,
                                 T *numbers,
                                 const T a,
                                 const T b
                                )
{
    const T scale = b - a;
    CUDA_1D_KERNEL_LOOP(n, size){
        numbers[n] = numbers[n] * scale + a;
    }
}

template <>
void shift_a_b<float, CUDAContext>(int size, float *x, float a, float b){
    shift_a_b_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x, a, b);
}

template <class T> __global__
void bernoulli_dist_kernel(const int size,
                           const T prob,
                           T *bernoulli
                          )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        bernoulli[n] = bernoulli[n] >= prob ? T(1) : T(0);
    }
}

template <>
void bernoulli_distribution<float, CUDAContext>(const int size,
                                                const float prob,
                                                float *bernoulli
                                               )
{
    curandGenerateUniform(cuda_context_v2::create()->get_curand_generator(), bernoulli, size);
    bernoulli_dist_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, prob, bernoulli);
}

#define DEFINE_BINARY_OP_KERNEL(OpName, Expr)      \
template <typename T>                              \
__global__ void OpName##_binary_op(const int size, \
                                   const T *a,     \
                                   const T *b,     \
                                   T *c){          \
    CUDA_1D_KERNEL_LOOP(n, size){                  \
        c[n] = a[n] Expr b[n];                     \
    }                                              \
}                                                  \
template <typename T> __global__                   \
void OpName##_val_binary_op(const int size,        \
                            const T val,           \
                            const T *a,            \
                            T *c){                 \
    CUDA_1D_KERNEL_LOOP(n, size){                  \
        c[n] = val Expr a[n];                      \
    }                                              \
}

DEFINE_BINARY_OP_KERNEL(Add, +)
DEFINE_BINARY_OP_KERNEL(Sub, -)
DEFINE_BINARY_OP_KERNEL(Mul, *)
DEFINE_BINARY_OP_KERNEL(Div, / )

#define DEFINE_CUDA_BINARY_OP(OpName)                  \
template <>                                            \
void OpName##Cuda<float>(const int size,               \
                         const float *a,               \
                         const float *b,               \
                         float *c                      \
                        )                              \
{                                                      \
    OpName##_binary_op<float><<<                       \
        CUDA_CONTEXT_GET_BLOCKS(size),                 \
        CUDA_CONTEXT_NUM_THREADS>>>(size, a, b, c);    \
}                                                      \
template <>                                            \
void OpName##ValCuda<float>(const int size,            \
                            const float val,           \
                            const float *a,            \
                            float *c)                  \
{                                                      \
    OpName##_val_binary_op<float><<<                   \
        CUDA_CONTEXT_GET_BLOCKS(size),                 \
        CUDA_CONTEXT_NUM_THREADS>>>(size, val, a, c);  \
}                                                      \
template <>                                            \
void OpName##Cuda<double>(                             \
                          const int size,              \
                          const double *a,             \
                          const double *b,             \
                          double *c                    \
                         )                             \
{                                                      \
    OpName##_binary_op<double><<<                      \
        CUDA_CONTEXT_GET_BLOCKS(size),                 \
        CUDA_CONTEXT_NUM_THREADS>>>(size, a, b, c);    \
}                                                      \
template <>                                            \
void OpName##ValCuda<double>(const int size,           \
                             const double val,         \
                             const double *a,          \
                             double *c                 \
                            )                          \
{                                                      \
    OpName##_val_binary_op<double><<<                  \
        CUDA_CONTEXT_GET_BLOCKS(size),                 \
        CUDA_CONTEXT_NUM_THREADS>>>(size, val, a, c);  \
}

DEFINE_CUDA_BINARY_OP(Add)
DEFINE_CUDA_BINARY_OP(Sub)
DEFINE_CUDA_BINARY_OP(Mul)
DEFINE_CUDA_BINARY_OP(Div)

} // end namespace math
} // end namespace mlfe
