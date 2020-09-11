#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <third_party/cub/cub/block/block_reduce.cuh>
#include "blas.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/core/device.h"

namespace mlfe{ namespace math{

template<>
void gemm<float, CUDAContext>(const bool trans_a,
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
                             )
{
    cublasOperation_t cuTransA =
        !trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        !trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (cublasSgemm(context->GetHandler(),
        cuTransB, cuTransA,
        n, m, k,
        &alpha, b_ptr, (!trans_b) ? n : k,
        a_ptr, (!trans_a) ? k : m,
        &beta, c_ptr, n) != CUBLAS_STATUS_SUCCESS) {
        throw std::string("gemm<float, CUDAContext> : cublasSgemm failed.");
    }
}

template<>
void gemm<double, CUDAContext>(const bool trans_a,
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
                              )
{
    cublasOperation_t cuTransA =
        !trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        !trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (cublasDgemm(context->GetHandler(),
        cuTransB, cuTransA,
        n, m, k,
        &alpha, b_ptr, (!trans_b) ? n : k,
        a_ptr, (!trans_a) ? k : m,
        &beta, c_ptr, n) != CUBLAS_STATUS_SUCCESS) {
        throw std::string("gemm<float, CUDAContext> : cublasDgemm failed.");
    }
}

template <>
void gemv<float, CUDAContext>(const bool trans_a,
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
                              )
{
    cublasOperation_t cuTransA = (!trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (cublasSgemv(
        context->GetHandler(),
        cuTransA, n, m,
        &alpha, a_ptr,
        n, b_ptr, 1,
        &beta, c_ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        throw std::string("gemv<float, CUDAContext> : cublasSgemv failed.");
    }
}

template <>
void gemv<double, CUDAContext>(const bool trans_a,
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
                              )
{
    cublasOperation_t cuTransA = (!trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (cublasDgemv(
        context->GetHandler(),
        cuTransA, n, m,
        &alpha, a_ptr,
        n, b_ptr, 1,
        &beta, c_ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        throw std::string("gemv<float, CUDAContext> : cublasDgemv failed.");
    }
}

} // end namespace math
} // end namespace mlfe
