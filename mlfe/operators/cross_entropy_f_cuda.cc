#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/activations.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"

namespace mlfe{
namespace algorithm_cuda{

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
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto logit_ptr = logit.device_data<T>();
        auto label_ptr = label.device_data<T>();
        auto loss_ptr = loss.mutable_device_data<T>();

        math::sigmoid_cross_entropy<T, CUDAContext>(
            m, n,
            logit_ptr,
            label_ptr,
            loss_ptr
            );
    }

private:
    Tensor logit;
    Tensor label;
    Tensor loss;
    int m, n;
};

REGIST_OP_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropy<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SigmoidCrossEntropyGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropyGrad(OpAlgoContext *oac) : OpAlgo(oac, "SigmoidCrossEntropyGradient"){
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

        math::sigmoid_cross_entropy_gradient<T, CUDAContext>(
            m, n,
            logit_ptr,
            label_ptr,
            loss_grad_ptr,
            logit_grad_ptr
            );
    }

private:
    Tensor logit;
    Tensor label;
    Tensor loss_grad;
    Tensor logit_grad;
    int m, n;
};

REGIST_OP_GRAD_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
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

        math::set<T, CUDAContext>(
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

        math::rowwise_max<T, CUDAContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        copy(logit.get_memory(), prob);

        math::gemm<T, CUDAContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, &ctx);

        math::exp<T, CUDAContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CUDAContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, &ctx);

        math::rowwise_normalize<T, CUDAContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy<T, CUDAContext>(m, n,
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
    CUDAContext ctx;
};

REGIST_OP_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device("CUDA")
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

        math::set<T, CUDAContext>(
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

        math::rowwise_max<T, CUDAContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        math::scal<T, CUDAContext>(
            m * n, T(1),
            x_ptr,
            prob_ptr
            );

        math::gemm<T, CUDAContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, &ctx);

        math::exp<T, CUDAContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CUDAContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, &ctx);

        math::rowwise_normalize<T, CUDAContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy_gradient<T, CUDAContext>(
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
    CUDAContext ctx;
};

REGIST_OP_GRAD_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabelGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // namespace algorithm_cuda
} // end namespace mlfe
