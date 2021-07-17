#include "mlfe/optimizers_v2/adam.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/optimizers.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

namespace {

template <typename T>
void adam_kernel(const int size,
          T *w,
          const T *dw,
          T *m_hist,
          T *v_hist,
          const T lr,
          const T beta1,
          const T beta2,
          const T eps)
{
    T correction = lr * sqrt(1.f - beta2) / (1.f - beta1);
    for(int n = 0; n < size; ++n){
        T g = dw[n];
        T mh = beta1 * m_hist[n] + (1.f - beta1) * g;
        T vh = beta2 * v_hist[n] + (1.f - beta2) * g * g;
        m_hist[n] = mh;
        v_hist[n] = vh;
        w[n] -= correction * mh / (sqrt(vh) + eps);
    }
}

template <typename T>
void adam_impl(Tensor x, Tensor dx, Tensor m_hist, Tensor v_hist, T lr, T beta1, T beta2, T eps){
    adam_kernel<T>(
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
