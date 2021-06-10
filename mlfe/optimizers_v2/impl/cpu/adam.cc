#include "mlfe/optimizers_v2/adam.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/optimizers.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

namespace {

template <typename T>
void adam_impl(Tensor x, Tensor dx, Tensor m_hist, Tensor v_hist, T lr, T beta1, T beta2, T eps){
    math::adam<T, CPUContext>(
        x.size(),
        x.mutable_device_data<T>(),
        dx.device_data<T>(),
        m_hist.mutable_device_data<T>(),
        v_hist.mutable_device_data<T>(),
        lr, beta1, beta2, eps);
}

} // namespace anonymous
} // namespace optimizers

REGIST_OP_KERNEL(adam, adam_fn_t, optimizers::adam_impl<float>);

} // namespace mlfe
