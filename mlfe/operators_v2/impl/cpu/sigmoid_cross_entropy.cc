#include "mlfe/operators_v2/sigmoid_cross_entropy.h"
#include "mlfe/core/op_kernel.h"

namespace mlfe{
namespace operators_v2{
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

        for (int t = 0; t < m; ++t){
            loss_ptr[t] = T(0);
            for (int u = 0; u < n; ++u){
                int idx = t * n + u;
                T a = logit_ptr[idx] * label_ptr[idx] - std::max(logit_ptr[idx], T(0));
                T b = std::log(T(1) + std::exp(-std::abs(logit_ptr[idx])));
                loss_ptr[t] += (a - b);
            }
            loss_ptr[t] = -loss_ptr[t] / static_cast<float>(n);
        }
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

        for(int b = 0; b < m; ++b){
            T dy_val = -dy_ptr[b] / T(n);
            for(int u = 0; u < n; ++u){
                int idx = b * n + u;
                T sig = T(1) / (T(1) + std::exp(-logit_ptr[idx]));
                logit_grad_ptr[idx] = (label_ptr[idx] - sig) * dy_val;
            }
        }
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
