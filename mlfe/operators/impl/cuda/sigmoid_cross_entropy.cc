#include "mlfe/operators/sigmoid_cross_entropy.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace operators{
namespace {

template <typename T>
struct sigmoid_cross_entropy_with_logits
{
    static void run(Tensor labels, Tensor logits, Tensor loss)
    {
        auto logit_ptr = logits.device_data<T>();
        auto label_ptr = labels.device_data<T>();
        auto loss_ptr = loss.mutable_device_data<T>();
        auto m = logits.shape()[0];
        auto n = logits.shape()[1];

        math::sigmoid_cross_entropy<T, CUDAContext>(
            m, n,
            logit_ptr,
            label_ptr,
            loss_ptr
            );
    }
};

template <typename T>
struct sigmoid_cross_entropy_with_logits_grad
{
    static void run(
        Tensor labels,
        Tensor logits,
        Tensor dy,
        Tensor d_logits
        )
    {
        auto logit_ptr = logits.device_data<T>();
        auto label_ptr = labels.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto logit_grad_ptr = d_logits.mutable_device_data<T>();
        auto m = logits.shape()[0];
        auto n = logits.shape()[1];

        math::sigmoid_cross_entropy_gradient<T, CUDAContext>(
            m, n,
            logit_ptr,
            label_ptr,
            dy_ptr,
            logit_grad_ptr
            );
    }
};

template <typename T>
void sigmoid_xent_fwd_impl(Tensor labels, Tensor logits, Tensor y){
    sigmoid_cross_entropy_with_logits<T>::run(labels, logits, y);
}

template <typename T>
void sigmoid_xent_bwd_impl(Tensor labels, Tensor logits, Tensor dy, Tensor dx){
    sigmoid_cross_entropy_with_logits_grad<T>::run(labels, logits, dy, dx);
}

} // namespace anonymous

REGIST_OP_KERNEL(
    sigmoid_xent_fwd,
    sigmoid_xent_fwd_fn_t,
    sigmoid_xent_fwd_impl<float>
    );

REGIST_OP_KERNEL(
    sigmoid_xent_bwd,
    sigmoid_xent_bwd_fn_t,
    sigmoid_xent_bwd_impl<float>
    );

} // namespace operators
} // namespace mlfe
