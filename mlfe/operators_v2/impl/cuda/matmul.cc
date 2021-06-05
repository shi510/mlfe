#include "mlfe/operators_v2/matmul.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/blas.h"

namespace mlfe{
namespace operators_v2{
namespace{

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
    CUDAContext c;
    math::gemm<T, CUDAContext>(
        transpose_a, transpose_b,
        m, n, k,
        T(1), a_ptr, a.shape()[1],
        b_ptr, b.shape()[1],
        T(0), y_ptr, y.shape()[1], &c);
}

} // namespace anonymous

REGIST_OP_KERNEL(matmul_fwd, matmul_fwd_fn_t, matmul_fwd_impl<float>);

} // namespace operators
} // namespace mlfe
