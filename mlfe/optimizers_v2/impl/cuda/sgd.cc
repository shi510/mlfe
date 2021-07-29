#include "mlfe/optimizers/sgd.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/optimizers/impl/cuda/kernel/sgd.h"

namespace mlfe{
namespace optimizers{
using namespace operators;

namespace {

template <typename T>
void sgd_impl(Tensor x, Tensor dx, Tensor mm_hist, T lr, T mm, T decay){
    cuda_kernel::gradient_descent_momentum<T>(
        x.size(),
        x.mutable_device_data<T>(),
        dx.device_data<T>(),
        mm_hist.mutable_device_data<T>(),
        lr,
        mm,
        decay
    );
}

} // namespace anonymous
} // namespace optimizers

REGIST_OP_KERNEL(sgd, sgd_fn_t, optimizers::sgd_impl<float>);

} // namespace mlfe
