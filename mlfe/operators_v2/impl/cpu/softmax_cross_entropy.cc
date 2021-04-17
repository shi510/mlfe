#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/blas.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/operators_v2/softmax_cross_entropy.h"

namespace mlfe{
namespace operators_v2{
namespace impl_cpu{

template <typename T>
struct softmax_cross_entropy_with_logits
{
    static void run(Tensor labels, Tensor logits, Tensor xent){
        int m = logits.shape()[0];
        int n = logits.shape()[1];
        int size = m * n;
        memory_ptr sm;
        memory_ptr rm;
        memory_ptr scal;
        memory_ptr prob;
        prob = create_memory(m * n * sizeof(T));
        sm = create_memory(n * sizeof(T));
        rm = create_memory(m * sizeof(T));
        scal = create_memory(m * sizeof(T));
        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm->mutable_device_data<T>()
            );

        auto x_ptr = logits.device_data<T>();
        auto t_ptr = labels.device_data<T>();
        auto loss_ptr = xent.mutable_device_data<T>();
        auto sm_ptr = sm->mutable_device_data<T>();
        auto rm_ptr = rm->mutable_device_data<T>();
        auto scal_ptr = scal->mutable_device_data<T>();
        auto prob_ptr = prob->mutable_device_data<T>();

        math::rowwise_max<T, CPUContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        copy(logits.get_memory(), prob);

        math::gemm<T, CPUContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, nullptr);

        math::exp<T, CPUContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CPUContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, nullptr);

        math::rowwise_normalize<T, CPUContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy<T, CPUContext>(m, n,
            prob_ptr,
            t_ptr,
            loss_ptr
            );
    }
};

template <typename T>
struct softmax_cross_entropy_with_logits_grad {

    static void run(Tensor labels, Tensor logits, Tensor dy, Tensor d_logits) {
        int m = logits.shape()[0];
        int n = logits.shape()[1];
        int size = m * n;
        memory_ptr sm;
        memory_ptr rm;
        memory_ptr scal;
        memory_ptr prob;
        prob = create_memory(m * n * sizeof(T));
        sm = create_memory(n * sizeof(T));
        rm = create_memory(m * sizeof(T));
        scal = create_memory(m * sizeof(T));
        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm->mutable_device_data<T>()
            );

        auto x_ptr = logits.device_data<T>();
        auto t_ptr = labels.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = d_logits.mutable_device_data<T>();
        auto sm_ptr = sm->mutable_device_data<T>();
        auto rm_ptr = rm->mutable_device_data<T>();
        auto scal_ptr = scal->mutable_device_data<T>();
        auto prob_ptr = prob->mutable_device_data<T>();

        math::rowwise_max<T, CPUContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        math::scal<T, CPUContext>(
            m * n, T(1),
            x_ptr,
            prob_ptr
            );

        math::gemm<T, CPUContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, nullptr);

        math::exp<T, CPUContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CPUContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, nullptr);

        math::rowwise_normalize<T, CPUContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy_gradient<T, CPUContext>(
            m, n,
            prob_ptr,
            t_ptr,
            dy_ptr,
            dx_ptr
            );
    }
};


template <typename T>
void softmax_xent_fwd_impl(Tensor labels, Tensor logits, Tensor y){
    softmax_cross_entropy_with_logits<T>::run(labels, logits, y);
}

REGIST_OP_KERNEL(
    softmax_xent_fwd,
    softmax_xent_fwd_fn_t,
    impl_cpu::softmax_xent_fwd_impl<float>
    );

template <typename T>
void softmax_xent_bwd_impl(Tensor labels, Tensor logits, Tensor dy, Tensor dx){
    softmax_cross_entropy_with_logits_grad<T>::run(labels, logits, dy, dx);
}

REGIST_OP_KERNEL(
    softmax_xent_bwd,
    softmax_xent_bwd_fn_t,
    impl_cpu::softmax_xent_bwd_impl<float>
    );

} // namespace impl_cpu
} // namespace op_kernels_cpu
} // namespace mlfe
