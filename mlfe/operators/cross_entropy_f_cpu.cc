#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cpu_context.h"
#include <cmath>
#include <algorithm>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class SigmoidCrossEntropy : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropy(OpAlgoContext *oac) : OpAlgo(oac, "SigmoidCrossEntropy"){
        loss = oac->get_output(0);
        logit = loss.get_children()[0];
        label = loss.get_children()[1];

        m = logit.shape()[0];
        n = logit.shape()[1];
        size = m * n;
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto logit_ptr = logit.device_data<T>();
        auto label_ptr = label.device_data<T>();
        auto loss_ptr = loss.mutable_device_data<T>();

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

private:
    Tensor logit;
    Tensor label;
    Tensor loss;
    int m, n;
    int size;
};

REGIST_OP_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropy<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SigmoidCrossEntropyGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropyGrad(OpAlgoContext *oac) : OpAlgo(oac){
        logit_grad = oac->get_output(0);
        logit = logit_grad.get_children()[0];
        label = logit_grad.get_children()[1];
        loss_grad = logit_grad.get_children()[3];
        m = logit.shape()[0];
        n = logit.shape()[1];
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto logit_ptr = logit.device_data<T>();
        auto label_ptr = label.device_data<T>();
        auto loss_grad_ptr = loss_grad.device_data<T>();
        auto logit_grad_ptr = logit_grad.mutable_device_data<T>();

        for(int b = 0; b < m; ++b){
            T dy_val = -loss_grad_ptr[b] / T(n);
            for(int u = 0; u < n; ++u){
                int idx = b * n + u;
                T sig = T(1) / (T(1) + std::exp(-logit_ptr[idx]));
                logit_grad_ptr[idx] = (label_ptr[idx] - sig) * dy_val;
            }
        }
    }

private:
    Tensor logit;
    Tensor label;
    Tensor loss_grad;
    Tensor logit_grad;
    int m, n;
    int size;
};

REGIST_OP_GRAD_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropyGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SoftmaxCrossEntropyWithLabel : public OpAlgo{
    using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabel(OpAlgoContext *oac) 
        : OpAlgo(oac, "SoftmaxCrossEntropyWithLabel"){
        loss = oac->get_output(0);
        logit = loss.get_children()[0];
        label = loss.get_children()[1];
        m = logit.shape()[0];
        n = logit.shape()[1];
        size = m * n;

        prob = create_memory(m * n * Tp::size);
        sm = create_memory(n * Tp::size);
        rm = create_memory(m * Tp::size);
        scal = create_memory(m * Tp::size);

        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm->mutable_device_data<T>()
            );
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = logit.device_data<T>();
        auto t_ptr = label.device_data<T>();
        auto loss_ptr = loss.mutable_device_data<T>();
        auto sm_ptr = sm->mutable_device_data<T>();
        auto rm_ptr = rm->mutable_device_data<T>();
        auto scal_ptr = scal->mutable_device_data<T>();
        auto prob_ptr = prob->mutable_device_data<T>();

        math::rowwise_max<T, CPUContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        copy(logit.get_memory(), prob);

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
private:
    Tensor logit;
    Tensor label;
    Tensor loss;
    memory_ptr sm;
    memory_ptr rm;
    memory_ptr scal;
    memory_ptr prob;
    int m, n;
    int size;
};

REGIST_OP_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabel<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SoftmaxCrossEntropyWithLabelGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabelGrad(OpAlgoContext *oac) : OpAlgo(oac){
        logit_grad = oac->get_output(0);
        logit = logit_grad.get_children()[0];
        label = logit_grad.get_children()[1];
        loss_grad = logit_grad.get_children()[3];
        m = logit.shape()[0];
        n = logit.shape()[1];
        size = m * n;

        prob = create_memory(m * n * Tp::size);
        sm = create_memory(n * Tp::size);
        rm = create_memory(m * Tp::size);
        scal = create_memory(m * Tp::size);

        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm->mutable_device_data<T>()
            );
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = logit.device_data<T>();
        auto t_ptr = label.device_data<T>();
        auto dy_ptr = loss_grad.mutable_device_data<T>();
        auto dx_ptr = logit_grad.mutable_device_data<T>();
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

private:
    Tensor logit;
    Tensor label;
    Tensor loss_grad;
    Tensor logit_grad;
    memory_ptr sm;
    memory_ptr rm;
    memory_ptr scal;
    memory_ptr prob;
    int m, n;
    int size;
};

REGIST_OP_GRAD_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabelGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
