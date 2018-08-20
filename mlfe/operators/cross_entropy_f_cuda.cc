#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/activations.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"

namespace mlfe{ namespace algorithm_cuda{

template <class Dev, class Tp>
class SigmoidCrossEntropy : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropy(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        t = oac->GetVar("Target");
        loss = oac->GetVar("Loss");

        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto loss_ptr = loss->Data<T>();

        math::sigmoid_cross_entropy<T, CUDAContext>(
            m, n,
            x_ptr,
            t_ptr,
            loss_ptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *loss;
    int m, n;
    int size;
};

REGIST_OP_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropy<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SigmoidCrossEntropyGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropyGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        t = oac->GetVar("Target");
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        m = x->Shape()[0];
        n = x->Shape()[1];
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        math::sigmoid_cross_entropy_gradient<T, CUDAContext>(
            m, n,
            x_ptr,
            t_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *dy;
    TensorMemRef *dx;
    int m, n;
};

REGIST_OP_GRAD_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropyGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SoftmaxCrossEntropyWithLabel : public OpAlgo{
using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabel(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        t = oac->GetVar("Target");
        loss = oac->GetVar("Loss");

        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;

        prob = Device::Select<Dev>();
        prob.Allocate(m * n * Tp::size);

        sm = Device::Select<Dev>();
        sm.Allocate(n * Tp::size);

        rm = Device::Select<Dev>();
        rm.Allocate(m * Tp::size);

        scal = Device::Select<Dev>();
        scal.Allocate(m * Tp::size);

        math::set<T, CUDAContext>(
            n,
            static_cast<T>(1),
            sm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto loss_ptr = loss->Data<T>();
        auto sm_ptr = sm.Data<T>();
        auto rm_ptr = rm.Data<T>();
        auto scal_ptr = scal.Data<T>();
        auto prob_ptr = prob.Data<T>();

        math::rowwise_max<T, CUDAContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        Device::Copy<Device::CUDA, Device::CUDA>(x->GetDevice(), prob);

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
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *loss;
    Device sm;
    Device rm;
    Device scal;
    Device prob;
    int m, n;
    int size;
    CUDAContext ctx;
};

REGIST_OP_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabel<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SoftmaxCrossEntropyWithLabelGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabelGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        t = oac->GetVar("Target");
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;

        prob = Device::Select<Dev>();
        prob.Allocate(m * n * Tp::size);

        sm = Device::Select<Dev>();
        sm.Allocate(n * Tp::size);

        rm = Device::Select<Dev>();
        rm.Allocate(m * Tp::size);

        scal = Device::Select<Dev>();
        scal.Allocate(m * Tp::size);

        math::set<T, CUDAContext>(
            n,
            static_cast<T>(1),
            sm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto sm_ptr = sm.Data<T>();
        auto rm_ptr = rm.Data<T>();
        auto scal_ptr = scal.Data<T>();
        auto prob_ptr = prob.Data<T>();

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
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *dy;
    TensorMemRef *dx;
    Device sm;
    Device rm;
    Device scal;
    Device prob;
    int m, n;
    int size;
    CUDAContext ctx;
};

REGIST_OP_GRAD_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabelGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // namespace algorithm_cuda
} // end namespace mlfe
