#include "mlfe/operators_v2/matmul.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace{

template<typename T>
void gemm(
    const bool trans_a,
    const bool trans_b,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T *a_ptr,
    const int lda,
    const T *b_ptr,
    const int ldb,
    const T beta,
    T *c_ptr,
    const int ldc
    )
{
    auto cuTransA = !trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto cuTransB = !trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto hdl = cuda_context_v2::create()->get_cublas_handle();
    auto status = cublasSgemm(
        hdl,
        cuTransB, cuTransA,
        n, m, k,
        &alpha, b_ptr, (!trans_b) ? n : k,
        a_ptr, (!trans_a) ? k : m,
        &beta, c_ptr, n);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout<<"[Warnning cublas] cublasSgemm failed."<<std::endl;
    }
}

template <typename T>
void matmul_fwd_impl(
    Tensor a,
    Tensor b,
    Tensor y,
    bool transpose_a,
    bool transpose_b
    )
{
    int32_t m = transpose_a ? a.shape()[1] : a.shape()[0];
    int32_t n = transpose_b ? b.shape()[0] : b.shape()[1];
    int32_t k = transpose_a ? a.shape()[0] : a.shape()[1];
    auto a_ptr = a.device_data<T>();
    auto b_ptr = b.device_data<T>();
    auto y_ptr = y.mutable_device_data<T>();

    gemm<T>(transpose_a, transpose_b,
        m, n, k,
        T(1), a_ptr, a.shape()[1],
        b_ptr, b.shape()[1],
        T(0), y_ptr, y.shape()[1]);
}

} // namespace anonymous

REGIST_OP_KERNEL(matmul_fwd, matmul_fwd_fn_t, matmul_fwd_impl<float>);

} // namespace operators
} // namespace mlfe
