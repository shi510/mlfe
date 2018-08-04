#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "blas.h"
#include "../device_context/cuda_context.h"

namespace mlfe{ namespace math{

template<>
void gemm<float, CUDAContext>(
                              const bool trans_a,
                              const bool trans_b,
                              const int m,
                              const int n,
                              const int k,
                              const float alpha,
                              const float *a_ptr,
                              const int lda,
                              const float *b_ptr,
                              const int ldb,
                              const float beta,
                              float *c_ptr,
                              const int ldc,
                              CUDAContext *context
                              ){
    cublasOperation_t cuTransA =
        !trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        !trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (cublasSgemm(context->GetHandler(),
        cuTransB, cuTransA,
        n, m, k,
        &alpha, b_ptr, (!trans_b) ? n : k,
        a_ptr, (!trans_a) ? k : m,
        &beta, c_ptr, n) != cudaSuccess) {
        throw std::string("gemm<float, CUDAContext> : cublasSgemm failed.");
    }
}

template<>
void gemm<double, CUDAContext>(
                                const bool trans_a,
                                const bool trans_b,
                                const int m,
                                const int n,
                                const int k,
                                const double alpha,
                                const double *a_ptr,
                                const int lda,
                                const double *b_ptr,
                                const int ldb,
                                const double beta,
                                double *c_ptr,
                                const int ldc,
                                CUDAContext *context
                                ){
    cublasOperation_t cuTransA =
        !trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        !trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (cublasDgemm(context->GetHandler(),
        cuTransB, cuTransA,
        n, m, k,
        &alpha, b_ptr, (!trans_b) ? n : k,
        a_ptr, (!trans_a) ? k : m,
        &beta, c_ptr, n) != cudaSuccess) {
        throw std::string("gemm<float, CUDAContext> : cublasDgemm failed.");
    }
}

template <>
void gemv<float, CUDAContext>(
                              const bool trans_a,
                              const int m,
                              const int n,
                              const float alpha,
                              const float *a_ptr,
                              const int lda,
                              const float *b_ptr,
                              const float beta,
                              float *c_ptr,
                              const int ldc,
                              CUDAContext *context
                              ){
    cublasOperation_t cuTransA = (!trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (cublasSgemv(
        context->GetHandler(),
        cuTransA, n, m,
        &alpha, a_ptr,
        n, b_ptr, 1,
        &beta, c_ptr, 1) != cudaSuccess) {
        throw std::string("gemv<float, CUDAContext> : cublasSgemv failed.");
    }
}

template <>
void gemv<double, CUDAContext>(
                                const bool trans_a,
                                const int m,
                                const int n,
                                const double alpha,
                                const double *a_ptr,
                                const int lda,
                                const double *b_ptr,
                                const double beta,
                                double *c_ptr,
                                const int ldc,
                                CUDAContext *context
                                ){
    cublasOperation_t cuTransA = (!trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (cublasDgemv(
        context->GetHandler(),
        cuTransA, n, m,
        &alpha, a_ptr,
        n, b_ptr, 1,
        &beta, c_ptr, 1) != cudaSuccess) {
        throw std::string("gemv<float, CUDAContext> : cublasDgemv failed.");
    }
}

template <class DataType> __global__
void rowwise_max_kernel(
                        const int rows, const int cols,
                        const DataType *data, DataType *out
                        ){
    typedef cub::BlockReduce<float, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
        DataType maxval = static_cast<DataType>(-FLT_MAX);
        for (int colIndex = threadIdx.x; colIndex < cols; colIndex += blockDim.x) {
            maxval = max(data[rowIndex * cols + colIndex], maxval);
        }
        maxval = BlockReduce(temp_storage).Reduce(maxval, cub::Max());
        if (threadIdx.x == 0) {
            out[rowIndex] = maxval;
        }
        __syncthreads();
    }
}

template <> void
rowwise_max<float, CUDAContext>(
                                const int m, const int n,
                                const float *a_ptr,
                                float *b_ptr
                                ){
    rowwise_max_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, a_ptr, b_ptr);
}

template <> void
rowwise_max<double, CUDAContext>(
                                 const int m, const int n,
                                 const double *a_ptr,
                                 double *b_ptr
                                 ){
    rowwise_max_kernel<double><<<
        CUDA_CONTEXT_GET_BLOCKS(n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, a_ptr, b_ptr);
}

template <class DT>
__global__ void rowwise_normalize_kernel(
    const int nthreads,
    const int D,
    const DT* scales,
    DT* out) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int n = index / D;
        out[index] /= scales[n];
    }
}

template <> void
rowwise_normalize<float, CUDAContext>(
                                      const int m, const int n,
                                      const float *scaler_ptr,
                                      float *norm_dest
                                      ){
    rowwise_normalize_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m * n, n, scaler_ptr, norm_dest);
}

template <> void
rowwise_normalize<double, CUDAContext>(
                                        const int m, const int n,
                                        const double *scaler_ptr,
                                        double *norm_dest
                                        ){
    rowwise_normalize_kernel<double><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m * n, n, scaler_ptr, norm_dest);
}

template <class DataType> __global__
void ProbCrossEntropyKernel(
                            const int N,
                            const int D,
                            const DataType *Pdata,
                            const DataType *labeldata,
                            DataType* Ydata
                            ){
    typedef cub::BlockReduce<float, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        DataType sum = static_cast<DataType>(0);
        DataType total_prob = static_cast<DataType>(0);
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            int idx = i * D + j;
            total_prob += labeldata[idx];
            sum += -log(max(Pdata[idx],
                                          static_cast<DataType>(1e-20))
                                          ) * labeldata[idx];
        }

        DataType tot = BlockReduce(temp_storage).Sum(sum);
        __syncthreads();
        DataType total_prob_sum = BlockReduce(temp_storage).Sum(total_prob);
        if (threadIdx.x == 0) {
            Ydata[i] = tot;
            // Sanity check
            //CUDA_KERNEL_ASSERT(abs(1.0 - total_prob_sum) < 1e-5f);
        }
        __syncthreads();
    }
}

template <> void
cross_entropy<float, CUDAContext>(
                                  const int m, const int n,
                                  const float *prob_ptr,
                                  const float *label_ptr,
                                  float *loss_ptr
                                  ){
  ProbCrossEntropyKernel<float><<<
      CUDA_CONTEXT_GET_BLOCKS(m * n),
      CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr);
}

template <> void
cross_entropy<double, CUDAContext>(
                                    const int m, const int n,
                                    const double *prob_ptr,
                                    const double *label_ptr,
                                    double *loss_ptr
                                    ){
  ProbCrossEntropyKernel<double><<<
      CUDA_CONTEXT_GET_BLOCKS(m * n),
      CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr);
}

template <class DataType> __global__
void CrossEntropyGradientKernel(
                                const int N,
                                const int D,
                                const DataType* Pdata,
                                const DataType* labeldata,
                                const DataType* lossdata,
                                DataType* dXdata
                                ){
    DataType avg = lossdata[0] / static_cast<DataType>(N);
    CUDA_1D_KERNEL_LOOP(idx, N * D){
        dXdata[idx] = (Pdata[idx] - labeldata[idx]) * avg;
    }
}

template <> void
cross_entropy_gradients<float, CUDAContext>(
                                            const int m, const int n,
                                            const float *prob_ptr,
                                            const float *label_ptr,
                                            const float *loss_ptr,
                                            float *dx_ptr
                                            ){
    CrossEntropyGradientKernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr, dx_ptr);
}

template <> void
cross_entropy_gradients<double, CUDAContext>(
                                            const int m, const int n,
                                            const double *prob_ptr,
                                            const double *label_ptr,
                                            const double *loss_ptr,
                                            double *dx_ptr
                                            ){
    CrossEntropyGradientKernel<double><<<
        CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr, dx_ptr);
}

template <class DataType> __global__
void exp_kernel(
                const int size,
                const DataType *x,
                DataType *y
                ){
    CUDA_1D_KERNEL_LOOP(index, size) {
        y[index] = std::exp(x[index]);
    }
}

template<>
void exp<float, CUDAContext>(
                            const int size,
                            const float *x_ptr,
                            float *y_ptr){
    exp_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, y_ptr);
}

template<>
void exp<double, CUDAContext>(
                             const int size,
                             const double *x_ptr,
                             double *y_ptr){
    exp_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, y_ptr);
}

template <class DataType> __global__
void axpy_kernel(
                  const int size,
                  const DataType a,
                  const DataType *x,
                  DataType *y
                  ){
    CUDA_1D_KERNEL_LOOP(index, size) {
        y[index] = a * x[index] + y[index];
    }
}

template<> void
axpy<float, CUDAContext>(
                          int size,
                          const float alpha,
                          const float *x_ptr,
                          float *y_ptr
                          ){
    axpy_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, alpha, x_ptr, y_ptr);
}

template<> void
axpy<double, CUDAContext>(
                          int size,
                          const double alpha,
                          const double *x_ptr,
                          double *y_ptr
                          ){
    axpy_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, alpha, x_ptr, y_ptr);
}

template <class DataType> __global__
void scale_kernel(
                  const int size,
                  const DataType *x,
                  const DataType a,
                  DataType *y
                  ){
    CUDA_1D_KERNEL_LOOP(index, size) {
        y[index] = a * x[index];
    }
}

template <> void
scal<float, CUDAContext>(
                          const int size,
                          const float alpha,
                          const float *x_ptr,
                          float *y_ptr
                          ){
    scale_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, alpha, y_ptr);
}

template <> void
scal<double, CUDAContext>(
                          const int size,
                          const double alpha,
                          const double *x_ptr,
                          double *y_ptr
                          ){
    scale_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x_ptr, alpha, y_ptr);
}

template <typename DataType> __global__
void SumKernel(
                const int N,
                const DataType* X,
                DataType* Y
                ){
    const int idx = threadIdx.x;
    __shared__ float reduction_buffer[128];

    reduction_buffer[idx] = 0;

    // A multilevel reduction.
    // N -> 128
    for (int i = idx; i < N; i += 128) {
        reduction_buffer[idx] += static_cast<float>(X[i]);
    }
    __syncthreads();
    // 128 -> 32
    if (idx < 32) {
        reduction_buffer[idx] +=
            reduction_buffer[idx + 32] +
            reduction_buffer[idx + 64] +
            reduction_buffer[idx + 96];
    }
    __syncthreads();
    // 32 -> 1
    if (idx == 0) {
        float tmp = 0;
        for (int i = 0; i < 32; ++i) {
            tmp += reduction_buffer[i];
        }
        *Y = static_cast<DataType>(tmp);
    }
}

template <>
void sum<float, CUDAContext>(
    const int size,
    const float *x_ptr,
    float *y_ptr){
    SumKernel<float><<<1, 128>>>(size, x_ptr, y_ptr);
}

template <>
void sum<double, CUDAContext>(
    const int size,
    const double *x_ptr,
    double *y_ptr){
    SumKernel<double><<<1, 128>>>(size, x_ptr, y_ptr);
}

template <class DataType> __global__
void set_kernel(
    const int size,
    const DataType val,
    DataType *x
    ){
    CUDA_1D_KERNEL_LOOP(index, size) {
        x[index] = val;
    }
}

template<>
void set<float, CUDAContext>(
    const int size,
    const float val,
    float *x_ptr
    ){
    set_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size), CUDA_CONTEXT_NUM_THREADS>>>(size, val, x_ptr);
}

template<>
void set<double, CUDAContext>(
    const int size,
    const double val,
    double *x_ptr
    ){
    set_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(size), CUDA_CONTEXT_NUM_THREADS>>>(size, val, x_ptr);
}

} /* math */
} /* mlfe */
