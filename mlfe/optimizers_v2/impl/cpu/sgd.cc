#include "mlfe/optimizers_v2/sgd.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/optimizers.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

namespace {

template <typename T>
void gradient_descent_momentum_kernel(const int size,
                               T *w,
                               const T *dw,
                               T *w_momentum,
                               const T lr,
                               const T momentum,
                               const T decay)
{
    for(int n = 0; n < size; ++n){
        w_momentum[n] = momentum * w_momentum[n];
        w_momentum[n] += lr * (dw[n] + decay * w[n]);
        w[n] -= w_momentum[n];
    }
}

template <typename T>
void sgd_impl(Tensor x, Tensor dx, Tensor mm_hist, T lr, T mm, T decay){
    gradient_descent_momentum_kernel<T>(
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
