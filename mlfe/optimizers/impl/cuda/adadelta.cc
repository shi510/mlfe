#include "mlfe/optimizers/adadelta.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/optimizers/impl/cuda/kernel/adadelta.h"

namespace mlfe{
namespace optimizers{
using namespace operators;

namespace {

template <typename T>
void adadelta_impl(Tensor x, Tensor dx, Tensor grad_hist, Tensor acc_hist, T lr, T momentum, T eps){
    auto size = x.size();
    auto x_ptr = x.mutable_device_data<T>();
    auto dx_ptr = dx.device_data<T>();
    auto grad_hist_ptr = grad_hist.mutable_device_data<T>();
    auto acc_hist_ptr = acc_hist.mutable_device_data<T>();
    cuda_kernel::adadelta<T>(
        size,
        x_ptr,
        dx_ptr,
        grad_hist_ptr,
        acc_hist_ptr,
        lr, momentum, eps);
}

} // namespace anonymous
} // namespace optimizers

REGIST_OP_KERNEL(adadelta, adadelta_fn_t, optimizers::adadelta_impl<float>);

} // namespace mlfe
