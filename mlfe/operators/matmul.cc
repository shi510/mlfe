#include "mlfe/operators/matmul.h"

namespace mlfe{
namespace operators{

Tensor matmul(Tensor a, Tensor b, bool transpose_a, bool transpose_b)
{
    int32_t m = transpose_a ? a.shape()[1] : a.shape()[0];
    int32_t n = transpose_b ? b.shape()[0] : b.shape()[1];
    int32_t k = transpose_a ? a.shape()[0] : a.shape()[1];
    auto output = functional::create_variable({m, n});
    auto gm_a = [=](Tensor &dy){
        if (!transpose_a && !transpose_b) {
            matmul_fwd_kernel::fn(dy, b, a.grad(), false, true);
        }
        else if (!transpose_a && transpose_b) {
            matmul_fwd_kernel::fn(dy, b, a.grad(), false, false);
        }
        else if (transpose_a && !transpose_b) {
            matmul_fwd_kernel::fn(b, dy, a.grad(), false, true);
        }
        else if (transpose_a && transpose_b) {
            matmul_fwd_kernel::fn(b, dy, a.grad(), true, true);
        }
    };
    auto gm_b = [=](Tensor &dy){
        if (!transpose_a && !transpose_b) {
            matmul_fwd_kernel::fn(a, dy, b.grad(), true, false);
        }
        else if (!transpose_a && transpose_b) {
            matmul_fwd_kernel::fn(dy, a, b.grad(), true, false);
        }
        else if (transpose_a && !transpose_b) {
            matmul_fwd_kernel::fn(a, dy, b.grad(), false, false);
        }
        else if (transpose_a && transpose_b) {
            matmul_fwd_kernel::fn(dy, a, b.grad(), true, true);
        }
    };
    call<matmul_fwd_kernel>(
        marker::I(a, b),
        marker::O(output)(gm_a, gm_b),
        transpose_a, transpose_b
    );
    return output;
}

} // namespace operators
} // namespace mlfe
